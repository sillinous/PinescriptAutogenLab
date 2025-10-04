# PineScript Autogen Lab

End-to-end starter for an autonomous TradingView + FastAPI stack:
- 🔌 Webhook `/exec` with optional HMAC verification
- 📒 Order journal + CSV export
- 📊 Unified positions & orders (Alpaca + CCXT stubs)
- 💹 P&L summary endpoint
- 🤖 AutoTune (Optuna) with walk-forward gate + promotion to `/strategy/params/best`
- 🧪 A/B live shadow deployments (control vs. candidate) with eligibility rules
- 🖥️ React dashboard (Vite) with Easy control panel

## Quickstart

### Backend
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8080
```
(Optional) Set `PINELAB_DATA=/path/to/state` to persist best params.

### Frontend
```bash
cd frontend
npm i
npm run dev
# open http://localhost:5173
```

### TradingView Webhook
- URL: `http://localhost:8080/exec`
- Optional header: `X-Signature: hex(hmac_sha256(secret, raw_body))`
- Equities (Alpaca):
```json
{"ticker":"AAPL","side":"buy","notional":5}
```
- Crypto (CCXT):
```json
{"market":"crypto","symbol":"BTC/USDT","side":"buy","qty":0.001}
```

## Git push
```bash
git init
git remote add origin https://github.com/sillinous/PinescriptAutogenLab.git
git checkout -b main
git add .
git commit -m "Bootstrap PineScript Autogen Lab (backend+frontend+Optuna+A/B)"
git push -u origin main
```

## Notes
- Alpaca calls are direct HTTP; add keys via the UI or POST `/broker/alpaca/set-creds`.
- CCXT varies by exchange; symbol format e.g. `BTC/USDT`.
- AutoTune’s walk-forward is stubbed; wire it to your backtester when ready.
- A/B shadow uses mock stats — replace with real metrics from journal/fills or portfolio telemetry.
