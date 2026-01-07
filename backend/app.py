
# backend/app.py


import asyncio
import os
import re
import sqlite3
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database import init_db, DB_PATH
from backend.config import Config
from backend.monitoring.logger import api_logger
from backend.auth.dependencies import rate_limit_normal, rate_limit_generous

CRYPTO_BASE = "https://api.crypto.com/v2"

app = FastAPI(title="PineLab AI Trading Platform", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# ============================================================================
# Router Integration with robust error handling for missing dependencies
# ============================================================================
try:
    from backend.api_auth import router as auth_router
    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    print("[INFO] Authentication endpoints loaded successfully")
except (ModuleNotFoundError, ImportError) as e:
    print(f"[WARNING] Authentication endpoints not loaded: {e}")

try:
    from backend.api_ai import router as ai_router
    app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI Trading"])
    print("[INFO] AI Trading endpoints loaded successfully")
except (ModuleNotFoundError, ImportError) as e:
    print(f"[WARNING] AI endpoints not loaded: {e}")

try:
    from backend.api_deep_learning import router as deep_learning_router
    app.include_router(deep_learning_router, prefix="/api/v1/ai", tags=["Deep Learning"])
    print("[INFO] Phase 2 Deep Learning endpoints loaded successfully")
except (ModuleNotFoundError, ImportError) as e:
    print(f"[WARNING] Phase 2 Deep Learning endpoints not loaded: {e}")

try:
    from backend.api_trading import router as trading_router
    app.include_router(trading_router, prefix="/api/v1", tags=["Autonomous Trading"])
    print("[INFO] Autonomous Trading endpoints loaded successfully")
except (ModuleNotFoundError, ImportError) as e:
    print(f"[WARNING] Autonomous Trading endpoints not loaded: {e}")
# ============================================================================


# Initialize the database using the centralized function
init_db()

def normalize_symbol(sym: str) -> str:
    s = sym.upper().replace("-", "").replace(":", "").strip()
    if "_" in s:
        base, quote = s.split("_", 1)
    else:
        if s.endswith("USDT"):
            base, quote = s[:-4], "USDT"
        elif s.endswith("USDC"):
            base, quote = s[:-4], "USDC"
        elif s.endswith("BUSD"):
            base, quote = s[:-4], "BUSD"
        elif s.endswith("USD"):
            base, quote = s[:-3], "USD"
        else:
            if "_" in s:
                return s
            base, quote = s, "USDT"
    return f"{base}_{quote}"

async def crypto_get(client: httpx.AsyncClient, path: str, params: Dict = None):
    r = await client.get(f"{CRYPTO_BASE}{path}", params=params or {}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise HTTPException(status_code=502, detail=f"Crypto.com error: {data}")
    return data["result"]

class Candle(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/health/live")
async def health_live():
    """Liveness probe - always returns OK if server is running"""
    return {"status": "ok", "check": "liveness"}

@app.get("/health/ready")
async def health_ready():
    """Readiness probe - checks if app is ready to serve traffic"""
    try:
        # Check database connection
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "check": "readiness", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

@app.get("/symbols", dependencies=[Depends(rate_limit_generous)])
async def symbols():
    async with httpx.AsyncClient() as client:
        res = await crypto_get(client, "/public/get-ticker")
        tickers = res.get("data", [])
        names = sorted({t.get("i") or t.get("instrument_name") for t in tickers if (t.get("i") or t.get("instrument_name"))})
        compact = [n.replace("_", "") for n in names]
        return {"symbols": names, "aliases": compact}

@app.get("/price/{symbol}", dependencies=[Depends(rate_limit_normal)])
async def price(symbol: str):
    sym = normalize_symbol(symbol)
    async with httpx.AsyncClient() as client:
        res = await crypto_get(client, "/public/get-ticker", params={"instrument_name": sym})
        data = res.get("data", [])
        if not data:
            raise HTTPException(status_code=404, detail=f"No ticker for {sym}")
        t = data[0]
        last = float(t.get("k", 0) or 0)
        if not last:
            a = float(t.get("a", 0) or 0); b = float(t.get("b", 0) or 0)
            last = (a + b) / 2 if a and b else (a or b)
        return {
            "symbol": sym,
            "last": last,
            "ask": float(t.get("a", last) or last),
            "bid": float(t.get("b", last) or last),
            "timestamp": int(t.get("t", 0) or 0),
        }

INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1D": "1D",
}

@app.get("/candles/{symbol}", dependencies=[Depends(rate_limit_normal)])
async def candles(symbol: str, interval: str = "1m", limit: int = 200):
    api_logger.debug(f"Candles endpoint called: symbol={symbol}, interval={interval}, limit={limit}")
    sym = normalize_symbol(symbol)
    api_logger.debug(f"Normalized symbol: {sym}, timeframe: {interval}")
    tf = INTERVAL_MAP.get(interval, "1m")
    limit = max(1, min(1000, int(limit)))
    api_logger.debug(f"Final params: sym={sym}, tf={tf}, limit={limit}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT t,o,h,l,c,v FROM candles WHERE symbol=? AND timeframe=? ORDER BY t DESC LIMIT ?",
                (sym, tf, limit))
    rows = cur.fetchall()
    conn.close()

    if rows and len(rows) >= int(limit * 0.8):
        data = [{"t": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4], "v": r[5]} for r in rows][::-1]
        return {"symbol": sym, "timeframe": tf, "candles": data}

    async with httpx.AsyncClient() as client:
        res = await crypto_get(client, "/public/get-candlestick", params={"instrument_name": sym, "timeframe": tf})

        # Handle Crypto.com API response format
        # Response structure: {"data": [...], "instrument_name": "...", "interval": "..."}
        if isinstance(res, dict) and "data" in res:
            candle_data = res["data"]
        elif isinstance(res, list):
            # Fallback: if res is directly a list
            candle_data = res
        else:
            raise HTTPException(status_code=502, detail=f"Unexpected response format from API: {type(res)}")

        if not candle_data:
            raise HTTPException(status_code=404, detail=f"No candle data found for {sym}")

        all_c = [{
            "t": int(int(c["t"]) // 1000),
            "o": float(c["o"]),
            "h": float(c["h"]),
            "l": float(c["l"]),
            "c": float(c["c"]),
            "v": float(c["v"]),
        } for c in candle_data]

        all_c = all_c[-limit:]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        for c in all_c:
            cur.execute(
                "INSERT OR REPLACE INTO candles(symbol, timeframe, t, o, h, l, c, v) VALUES(?,?,?,?,?,?,?,?)",
                (sym, tf, c["t"], c["o"], c["h"], c["l"], c["c"], c["v"]),
            )
        conn.commit()
        conn.close()

        return {"symbol": sym, "timeframe": tf, "candles": all_c}

def _pct_change(vals):
    if not vals or len(vals) < 2: return 0.0
    start, end = vals[0], vals[-1]
    if start == 0: return 0.0
    return (end - start) / start * 100.0

@app.get("/ab/status", dependencies=[Depends(rate_limit_generous)])
async def ab_status(symbol_a: str = "BTC_USDT", symbol_b: str = "ETH_USDT"):
    """
    Compare momentum performance between two symbols.
    Users can select which symbols to compare via query parameters.
    """
    try:
        data_a = await candles(symbol_a, "1m", 120)
        data_b = await candles(symbol_b, "1m", 120)
    except Exception as e:
        api_logger.warning(f"Failed to fetch candles for A/B test: {e}")
        return {
            "test_name": f"{symbol_a} vs {symbol_b} (2h momentum)",
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "variant_a_winrate": 50.0,
            "variant_b_winrate": 50.0,
            "winner": "TIE",
            "error": "Unable to fetch market data"
        }

    ret_a = _pct_change([c["c"] for c in data_a["candles"]])
    ret_b = _pct_change([c["c"] for c in data_b["candles"]])

    if abs(ret_a - ret_b) < 0.01:
        winner = "TIE"
    else:
        winner = "A" if ret_a >= ret_b else "B"

    return {
        "test_name": f"{symbol_a} vs {symbol_b} (2h momentum)",
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "variant_a_winrate": round(max(0.0, min(100.0, 50 + ret_a / 2)), 2),
        "variant_b_winrate": round(max(0.0, min(100.0, 50 + ret_b / 2)), 2),
        "winner": winner,
        "returns": {
            "symbol_a": round(ret_a, 4),
            "symbol_b": round(ret_b, 4)
        }
    }

@app.get("/autotune/status", dependencies=[Depends(rate_limit_generous)])
async def autotune_status(symbol: str = "BTC_USDT"):
    """
    Auto-optimize SMA window for a given symbol.
    Users can select which symbol to analyze via query parameter.
    """
    try:
        data = await candles(symbol, "1m", 300)
    except Exception as e:
        api_logger.warning(f"Failed to fetch candles for autotune: {e}")
        return {
            "symbol": symbol,
            "progress": 0,
            "best_parameters": {"sma_window": None, "score": 0},
            "error": "Unable to fetch market data"
        }

    closes = [c["c"] for c in data["candles"]]
    best_w = None
    best_score = -1e9
    tested = 0
    for w in range(5, 41):
        if len(closes) < w + 2:
            continue
        rets = [(closes[i] - closes[i-1]) / (closes[i-1] or 1e-9) for i in range(1, len(closes))]
        window = rets[-w:]
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / max(1, len(window)-1)
        vol = (var ** 0.5) if var > 0 else 1e-6
        score = mean / vol
        tested += 1
        if score > best_score:
            best_score = score
            best_w = w
    progress = round(min(100.0, tested / 36 * 100.0), 1)
    return {
        "symbol": symbol,
        "progress": progress,
        "best_parameters": {"sma_window": best_w, "score": round(best_score, 4)}
    }

# Note: AI endpoints are now loaded from api_ai.py router with /api/v1/ai prefix
# No stub endpoints needed - real AI features available via router
