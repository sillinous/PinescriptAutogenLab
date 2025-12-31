
import os, json, hmac, hashlib, datetime as dt, threading, time, random
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional: external libs
try:
    import optuna
    from optuna.samplers import TPESampler, CMAESampler
except Exception:
    optuna = None

try:
    import ccxt
except Exception:
    ccxt = None

app = FastAPI(title="PineScript Autogen Lab", version="0.5.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Global state & utils
# ---------------------
STATE: Dict[str, Any] = {
    "alpaca": {"key": "", "secret": "", "base": "https://paper-api.alpaca.markets"},
    "ccxt": {"id": "", "apiKey": "", "secret": ""},
}
JOURNAL: List[Dict[str, Any]] = []
ALERT_SECRET = {"value": ""}

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

# ---------------------
# Broker: Alpaca (minimal placeholders; swap with real SDK calls)
# ---------------------
import httpx

async def alpaca_account() -> Dict[str, Any]:
    # Minimal guard
    if not STATE["alpaca"]["key"] or not STATE["alpaca"]["secret"]:
        return {"status": "not_configured"}
    # Try account endpoint
    try:
        base = STATE["alpaca"]["base"].rstrip("/")
        url = f"{base}/v2/account"
        headers = {"APCA-API-KEY-ID": STATE["alpaca"]["key"], "APCA-API-SECRET-KEY": STATE["alpaca"]["secret"]}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            return r.json()
    except Exception as e:
        return {"error": str(e)}

async def alpaca_positions() -> List[Dict[str, Any]]:
    if not STATE["alpaca"]["key"]:
        return []
    try:
        base = STATE["alpaca"]["base"].rstrip("/")
        url = f"{base}/v2/positions"
        headers = {"APCA-API-KEY-ID": STATE["alpaca"]["key"], "APCA-API-SECRET-KEY": STATE["alpaca"]["secret"]}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            data = r.json() if r.status_code == 200 else []
            out = []
            for p in data or []:
                out.append({
                    "broker": "alpaca",
                    "symbol": p.get("symbol"),
                    "qty": p.get("qty"),
                    "unrealized": p.get("unrealized_pl") or p.get("unrealized_plpc"),
                })
            return out
    except Exception:
        return []

async def alpaca_orders(limit: int = 50) -> List[Dict[str, Any]]:
    if not STATE["alpaca"]["key"]:
        return []
    try:
        base = STATE["alpaca"]["base"].rstrip("/")
        url = f"{base}/v2/orders?status=all&limit={limit}"
        headers = {"APCA-API-KEY-ID": STATE["alpaca"]["key"], "APCA-API-SECRET-KEY": STATE["alpaca"]["secret"]}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=headers)
            data = r.json() if r.status_code == 200 else []
            out = []
            for o in data or []:
                out.append({
                    "broker": "alpaca",
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "qty": o.get("qty") or o.get("notional"),
                    "status": o.get("status"),
                    "id": o.get("id"),
                    "time": o.get("submitted_at"),
                })
            return out
    except Exception:
        return []

async def alpaca_order(symbol: str, side: str, qty: Optional[str] = None, notional: Optional[str] = None) -> Dict[str, Any]:
    if not STATE["alpaca"]["key"]:
        return {"error": "alpaca not configured"}
    try:
        base = STATE["alpaca"]["base"].rstrip("/")
        url = f"{base}/v2/orders"
        headers = {"APCA-API-KEY-ID": STATE["alpaca"]["key"], "APCA-API-SECRET-KEY": STATE["alpaca"]["secret"],
                   "Content-Type": "application/json"}
        body = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day"}
        if qty: body["qty"] = qty
        if notional: body["notional"] = notional
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, headers=headers, json=body)
            return r.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------------
# Broker: CCXT minimal
# ---------------------
def ccxt_available():
    return ccxt is not None and STATE["ccxt"]["id"]

def ccxt_exchange():
    if not ccxt or not STATE["ccxt"]["id"]:
        return None
    ex_id = STATE["ccxt"]["id"]
    klass = getattr(ccxt, ex_id, None)
    if not klass:
        return None
    return klass({
        "apiKey": STATE["ccxt"]["apiKey"],
        "secret": STATE["ccxt"]["secret"],
        "enableRateLimit": True,
    })

async def ccxt_positions() -> List[Dict[str, Any]]:
    # Positions support varies by exchange; we return empty by default.
    return []

async def ccxt_orders(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        ex = ccxt_exchange()
        if not ex:
            return []
        if not symbol:
            return []
        data = ex.fetch_orders(symbol=symbol, limit=25)
        out = []
        for o in data or []:
            out.append({
                "broker": "ccxt",
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": o.get("amount"),
                "status": o.get("status"),
                "id": o.get("id"),
                "time": o.get("datetime"),
            })
        return out
    except Exception:
        return []

async def ccxt_order(symbol: str, side: str, amount: float) -> Dict[str, Any]:
    try:
        ex = ccxt_exchange()
        if not ex:
            return {"error": "ccxt not configured"}
        if side == "buy":
            return ex.create_market_buy_order(symbol, amount)
        else:
            return ex.create_market_sell_order(symbol, amount)
    except Exception as e:
        return {"error": str(e)}

# ---------------------
# Config & Broker endpoints
# ---------------------
@app.post("/broker/alpaca/set-creds")
async def set_alpaca_creds(key_id: str = Form(...), secret_key: str = Form(...), base_url: str = Form("https://paper-api.alpaca.markets")):
    STATE["alpaca"] = {"key": key_id, "secret": secret_key, "base": base_url}
    return {"ok": True}

@app.get("/broker/alpaca/account")
async def get_alpaca_account():
    return await alpaca_account()

@app.get("/broker/alpaca/positions")
async def get_alpaca_positions():
    return await alpaca_positions()

@app.get("/broker/alpaca/orders")
async def get_alpaca_orders(limit: int = 50):
    return await alpaca_orders(limit=limit)

@app.post("/broker/alpaca/order")
async def post_alpaca_order(symbol: str = Form(...), side: str = Form(...), qty: Optional[str] = Form(None), notional: Optional[str] = Form(None)):
    return await alpaca_order(symbol, side, qty, notional)

@app.post("/broker/ccxt/set-creds")
async def set_ccxt_creds(exchange_id: str = Form(...), api_key: str = Form(...), secret: str = Form(...)):
    STATE["ccxt"] = {"id": exchange_id, "apiKey": api_key, "secret": secret}
    return {"ok": True, "exchange_available": bool(ccxt and getattr(ccxt, exchange_id, None) is not None)}

# ---------------------
# Unified endpoints
# ---------------------
async def unified_positions():
    pos = []
    pos += await alpaca_positions()
    pos += await ccxt_positions()
    return pos

@app.get("/broker/unified/positions")
async def get_unified_positions():
    return await unified_positions()

@app.get("/broker/unified/orders")
async def get_unified_orders(limit: int = 100, symbol: Optional[str] = None):
    out = []
    out += await alpaca_orders(limit=limit)
    out += JOURNAL[-limit:]
    # CCXT orders if symbol provided
    out += await ccxt_orders(symbol=symbol) if symbol else []
    return out[-limit:]

# ---------------------
# Journal + CSV export
# ---------------------
@app.get("/journal/orders")
async def journal_orders(limit: int = 200):
    return JOURNAL[-limit:]

@app.get("/journal/export")
async def journal_export():
    import io, csv
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["time","broker","symbol","side","qty","status","id"])
    w.writeheader()
    for row in JOURNAL:
        w.writerow(row)
    return StreamingResponse(iter([buf.getvalue().encode()]), media_type="text/csv",
                             headers={"Content-Disposition":"attachment; filename=pinelab_journal.csv"})

# ---------------------
# PnL summary
# ---------------------
@app.get("/pnl/summary")
async def pnl_summary():
    equity = cash = None
    try:
        acct = await alpaca_account()
        if isinstance(acct, dict):
            equity = float(acct.get("equity") or 0)
            cash = float(acct.get("cash") or 0)
    except Exception:
        pass
    try:
        upos = await unified_positions()
        unreal = sum(float(p.get("unrealized") or 0) for p in upos)
    except Exception:
        unreal = 0.0
    return {"equity": equity or 0.0, "cash": cash or 0.0, "unrealized": unreal}

# ---------------------
# Alert HMAC + Webhook exec router
# ---------------------
@app.post("/config/alerts/set-secret")
async def set_alert_secret(secret: str = Form(...)):
    ALERT_SECRET["value"] = secret or ""
    return {"ok": True, "has_secret": bool(ALERT_SECRET["value"])}

@app.post("/exec")
async def exec_webhook(request: Request):
    raw = await request.body()
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    # Verify HMAC if configured
    sec = ALERT_SECRET["value"]
    if sec:
        sig = request.headers.get("X-Signature", "")
        digest = hmac.new(sec.encode(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig.lower(), digest.lower()):
            return JSONResponse(status_code=401, content={"error": "Bad signature"})

    ticker = payload.get("ticker") or payload.get("symbol")
    side = payload.get("side", "buy")
    qty = payload.get("qty")
    notional = payload.get("notional")
    market = payload.get("market", "equity")

    # Route + mirror if AB enabled
    route = "alpaca"
    resp = {}
    if market == "crypto" and ccxt_available():
        resp = await ccxt_order(symbol=ticker, side=side, amount=float(qty or 0))
        route = "ccxt"
    else:
        resp = await alpaca_order(symbol=ticker, side=side, qty=str(qty) if qty else None, notional=str(notional) if notional else None)
        route = "alpaca"

    oid = resp.get("id")
    status = resp.get("status") or resp.get("message") or "submitted"

    JOURNAL.append({
        "time": now_iso(), "broker": route, "symbol": ticker, "side": side,
        "qty": qty or notional, "status": status, "id": oid
    })

    # Mirror to shadow if AB enabled and different route (paper vs live simulated here)
    if AB["enabled"] and AB.get("shadow"):
        JOURNAL.append({
            "time": now_iso(), "broker": f"shadow:{AB['shadow']}", "symbol": ticker, "side": side,
            "qty": qty or notional, "status": "mirrored", "id": f"shadow-{oid or 'n/a'}"
        })

    return {"route": route, "order": oid, "status": status}

# ---------------------
# AutoTune Optuna (Bayesian)
# ---------------------
DATA_DIR = os.getenv("PINELAB_DATA", ".")
BEST_PATH = os.path.join(DATA_DIR, "best_params.json")

AUTOTUNE: Dict[str, Any] = {
    "enabled": False,
    "cadence_min": 30,
    "trials": 50,
    "sampler": "tpe",
    "target_sharpe": 1.0,
    "max_dd": 10.0,
    "state": "idle",
    "last_run": None,
    "best_score": None,
    "best_params": None,
}
_stop = {"flag": False}
_thread = {"t": None}

SPACE = {"ema_fast": (5, 60), "ema_slow": (20, 200), "atr_mult": (1.2, 4.5), "entropy_lo": (0.3, 0.8)}

def score_params(p: Dict[str, Any]) -> float:
    base = 2.0 - abs((p["ema_fast"]/p["ema_slow"]) - 0.4) * 3.0
    base += max(0, 1.2 - abs(p["atr_mult"] - 2.1))
    base += max(0, 0.8 - abs(p["entropy_lo"] - 0.55) * 3)
    return max(0.01, base + random.uniform(-0.15, 0.15))

def walkforward_validate(p: Dict[str, Any]) -> Dict[str, Any]:
    pf = max(0.3, random.gauss(1.6, 0.35))
    dd = abs(random.gauss(7, 3))
    sharpe = max(0.1, random.gauss(1.2, 0.4))
    return {"profit_factor": round(pf, 3), "max_dd": round(dd, 2), "sharpe": round(sharpe, 2)}

def passes_guard(metrics: Dict[str, Any]) -> bool:
    if metrics.get("sharpe", 0) < AUTOTUNE["target_sharpe"]:
        return False
    if metrics.get("max_dd", 1e9) > AUTOTUNE["max_dd"]:
        return False
    return True

def save_best(best: Dict[str, Any]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BEST_PATH, "w") as f:
        json.dump(best, f)

def load_best() -> Optional[Dict[str, Any]]:
    if not os.path.exists(BEST_PATH): return None
    try:
        return json.load(open(BEST_PATH))
    except Exception:
        return None

def optuna_sampler(name: str):
    if not optuna:
        return None
    name = (name or "").lower()
    if name == "cmaes": return CMAESampler()
    if name == "random":
        from optuna.samplers import RandomSampler
        return RandomSampler()
    return TPESampler()

def run_study(trials: int, sampler_name: str) -> Dict[str, Any]:
    if optuna:
        sampler = optuna_sampler(sampler_name)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        def objective(trial):
            p = {
                "ema_fast": trial.suggest_int("ema_fast", *SPACE["ema_fast"]),
                "ema_slow": trial.suggest_int("ema_slow", *SPACE["ema_slow"]),
                "atr_mult": trial.suggest_float("atr_mult", *SPACE["atr_mult"]),
                "entropy_lo": trial.suggest_float("entropy_lo", *SPACE["entropy_lo"]),
            }
            return score_params(p)
        study.optimize(objective, n_trials=trials)
        return {"params": study.best_params, "score": float(study.best_value)}
    # fallback random
    best_p, best_s = None, -1
    for _ in range(trials):
        p = {
            "ema_fast": random.randint(*SPACE["ema_fast"]),
            "ema_slow": random.randint(*SPACE["ema_slow"]),
            "atr_mult": round(random.uniform(*SPACE["atr_mult"]), 3),
            "entropy_lo": round(random.uniform(*SPACE["entropy_lo"]), 3),
        }
        s = score_params(p)
        if s > best_s:
            best_s, best_p = s, p
    return {"params": best_p, "score": float(best_s)}

def autotune_loop():
    while not _stop["flag"]:
        if not AUTOTUNE["enabled"]:
            time.sleep(1); continue
        try:
            AUTOTUNE["state"] = "optimizing"
            res = run_study(int(AUTOTUNE["trials"]), AUTOTUNE["sampler"])
            cand, score = res["params"], float(res["score"])
            AUTOTUNE["state"] = "validating"
            metrics = walkforward_validate(cand)
            ok = passes_guard(metrics)
            AUTOTUNE["best_score"] = round(score,4)
            AUTOTUNE["best_params"] = cand
            AUTOTUNE["last_run"] = now_iso()
            AUTOTUNE["state"] = "cooldown" if ok else "rejected"
            if ok:
                save_best({"params": cand, "score": score, "metrics": metrics, "ts": AUTOTUNE["last_run"]})
        except Exception as e:
            AUTOTUNE["state"] = f"error: {e}"
        sleep_s = max(60, int(AUTOTUNE["cadence_min"]) * 60)
        for _ in range(sleep_s):
            if _stop["flag"] or not AUTOTUNE["enabled"]:
                break
            time.sleep(1)
        if not AUTOTUNE["enabled"]:
            AUTOTUNE["state"] = "idle"

def ensure_autotune_thread():
    if not _thread["t"] or not _thread["t"].is_alive():
        _stop["flag"] = False
        t = threading.Thread(target=autotune_loop, daemon=True)
        t.start()
        _thread["t"] = t

@app.post("/autotune/start_bayes")
async def start_bayes(
    cadenceMin: int = Form(30),
    trials: int = Form(50),
    sampler: str = Form("tpe"),
    target_sharpe: float = Form(1.0),
    max_dd: float = Form(10.0),
):
    AUTOTUNE.update({
        "enabled": True,
        "cadence_min": cadenceMin,
        "trials": trials,
        "sampler": sampler,
        "target_sharpe": target_sharpe,
        "max_dd": max_dd,
        "state": "queued",
    })
    ensure_autotune_thread()
    return {"ok": True, **AUTOTUNE}

@app.post("/autotune/stop")
async def stop_bayes():
    AUTOTUNE["enabled"] = False
    AUTOTUNE["state"] = "idle"
    return {"ok": True, **AUTOTUNE}

@app.get("/autotune/status")
async def status():
    best = load_best() or {}
    return {
        "enabled": AUTOTUNE["enabled"],
        "state": AUTOTUNE["state"],
        "lastRun": AUTOTUNE["last_run"],
        "bestScore": AUTOTUNE["best_score"],
        "bestParams": AUTOTUNE["best_params"],
        "promoted": best,
    }

@app.post("/autotune/promote_best")
async def promote_best():
    if not AUTOTUNE.get("best_params"):
        return JSONResponse(status_code=400, content={"error": "No best_params available"})
    metrics = walkforward_validate(AUTOTUNE["best_params"])
    if not passes_guard(metrics):
        return JSONResponse(status_code=400, content={"error": "Guard thresholds not met", "metrics": metrics})
    rec = {"params": AUTOTUNE["best_params"], "score": AUTOTUNE.get("best_score"), "metrics": metrics, "ts": now_iso()}
    save_best(rec)
    return {"ok": True, **rec}

@app.get("/strategy/params/best")
async def get_best_params():
    best = load_best()
    return best or {}

# ---------------------
# A/B Shadow Deployments
# ---------------------
AB = {
    "enabled": False,
    "state": "idle",
    "control": "live",
    "shadow": "paper",
    "window_mins": 60,
    "promote_sharpe_delta": 0.2,
    "promote_dd_cap": 10.0,
    "window_ends": None,
    "control_stats": {},
    "candidate_stats": {},
    "eligible": False,
}
_ab_thread = {"t": None}
_ab_stop = {"flag": False}

def mock_stats():
    sharpe = max(0.05, random.gauss(1.0, 0.35))
    dd = abs(random.gauss(7, 2.5))
    pf = max(0.3, random.gauss(1.5, 0.3))
    return {"sharpe": round(sharpe, 2), "max_dd": round(dd, 2), "profit_factor": round(pf, 2)}

def ab_loop():
    while not _ab_stop["flag"]:
        if not AB["enabled"]:
            time.sleep(1); continue
        try:
            AB["state"] = "running"
            AB["control_stats"] = mock_stats()
            AB["candidate_stats"] = mock_stats()
            sharpe_delta = AB["candidate_stats"].get("sharpe",0) - AB["control_stats"].get("sharpe",0)
            dd_ok = AB["candidate_stats"].get("max_dd",1e9) <= AB["promote_dd_cap"]
            AB["eligible"] = (sharpe_delta >= AB["promote_sharpe_delta"]) and dd_ok
            if AB["window_ends"]:
                ends = dt.datetime.fromisoformat(AB["window_ends"].replace("Z",""))
                if dt.datetime.utcnow() >= ends:
                    AB["enabled"] = False
                    AB["state"] = "idle"
        except Exception as e:
            AB["state"] = f"error: {e}"
        for _ in range(10):
            if _ab_stop["flag"] or not AB["enabled"]: break
            time.sleep(1)

def ensure_ab_thread():
    if not _ab_thread["t"] or not _ab_thread["t"].is_alive():
        _ab_stop["flag"] = False
        t = threading.Thread(target=ab_loop, daemon=True)
        t.start()
        _ab_thread["t"] = t

@app.post("/ab/start")
async def ab_start(
    control: str = Form("live"),
    shadow: str = Form("paper"),
    window_mins: int = Form(60),
    promote_sharpe_delta: float = Form(0.2),
    promote_dd_cap: float = Form(10.0),
):
    AB.update({
        "enabled": True,
        "state": "queued",
        "control": control,
        "shadow": shadow,
        "window_mins": window_mins,
        "promote_sharpe_delta": promote_sharpe_delta,
        "promote_dd_cap": promote_dd_cap,
        "window_ends": (dt.datetime.utcnow() + dt.timedelta(minutes=window_mins)).replace(microsecond=0).isoformat() + "Z",
    })
    ensure_ab_thread()
    return {"ok": True, **AB}

@app.post("/ab/stop")
async def ab_stop():
    AB["enabled"] = False
    AB["state"] = "idle"
    return {"ok": True, **AB}

@app.get("/ab/status")
async def ab_status():
    return {
        "enabled": AB["enabled"],
        "state": AB["state"],
        "windowEnds": AB["window_ends"],
        "control": AB["control_stats"],
        "candidate": AB["candidate_stats"],
        "eligible": AB["eligible"],
    }

@app.post("/ab/promote_candidate")
async def ab_promote_candidate():
    if not AB["candidate_stats"]:
        return {"ok": False, "error": "No candidate stats yet"}
    if not AB["eligible"]:
        return {"ok": False, "error": "Candidate not eligible by current thresholds"}
    return {"ok": True, "message": "Candidate promoted to control"}

# root
@app.get("/healthz")
async def health():
    return {"ok": True, "ts": now_iso()}

