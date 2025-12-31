
# PineScript Autogen Lab — Crypto.com (Public, Production-Ready)

## Quick Start

### 1) Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

### 2) Frontend
```bash
cd ../frontend
npm install
npm run dev
# open http://localhost:5173
```

## Endpoints
- `/symbols`
- `/price/{symbol}`
- `/candles/{symbol}?interval=1m&limit=200`
- `/summary`
- `/ab/status`
- `/autotune/status`
- `/healthz`

## Notes
- Live Crypto.com public API (no mock data).
- SQLite cache auto-created at `backend/cache.db`.
