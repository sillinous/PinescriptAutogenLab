
import asyncio
import os
import re
import sqlite3
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CRYPTO_BASE = "https://api.crypto.com/v2"

app = FastAPI(title="PineLab Crypto — Public Data API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "cache.db")

def db_init():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS candles(
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        t INTEGER NOT NULL,
        o REAL NOT NULL,
        h REAL NOT NULL,
        l REAL NOT NULL,
        c REAL NOT NULL,
        v REAL NOT NULL,
        PRIMARY KEY(symbol, timeframe, t)
    )
    """)
    conn.commit()
    conn.close()

db_init()

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

@app.get("/symbols")
async def symbols():
    async with httpx.AsyncClient() as client:
        res = await crypto_get(client, "/public/get-ticker")
        tickers = res.get("data", [])
        names = sorted({t.get("i") or t.get("instrument_name") for t in tickers if (t.get("i") or t.get("instrument_name"))})
        compact = [n.replace("_", "") for n in names]
        return {"symbols": names, "aliases": compact}

@app.get("/price/{symbol}")
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

@app.get("/candles/{symbol}")
async def candles(symbol: str, interval: str = "1m", limit: int = 200):
    sym = normalize_symbol(symbol)
    tf = INTERVAL_MAP.get(interval, "1m")
    limit = max(1, min(1000, int(limit)))

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
        arr = res.get("data", {}).get("candlestick", [])
        if not arr:
            raise HTTPException(status_code=404, detail=f"No candles for {sym}")
        all_c = [{
            "t": int(int(c["t"]) // 1000),
            "o": float(c["o"]),
            "h": float(c["h"]),
            "l": float(c["l"]),
            "c": float(c["c"]),
            "v": float(c["v"]),
        } for c in arr[0].get("data", [])]

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

@app.get("/ab/status")
async def ab_status():
    btc = await candles("BTC_USDT", "1m", 120)
    eth = await candles("ETH_USDT", "1m", 120)
    btc_ret = _pct_change([c["c"] for c in btc["candles"]])
    eth_ret = _pct_change([c["c"] for c in eth["candles"]])
    winner = "A" if btc_ret >= eth_ret else "B"
    return {
        "test_name": "BTC vs ETH (1h momentum)",
        "variant_a_winrate": round(max(0.0, min(100.0, 50 + btc_ret / 2)), 2),
        "variant_b_winrate": round(max(0.0, min(100.0, 50 + eth_ret / 2)), 2),
        "winner": winner,
    }

@app.get("/autotune/status")
async def autotune_status():
    data = await candles("BTC_USDT", "1m", 300)
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
    return {"progress": progress, "best_parameters": {"sma_window": best_w, "score": round(best_score, 4)}}
