# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import random

app = FastAPI(title="PineLab Backend API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data models
class PnLSummary(BaseModel):
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float

class AutoTuneStatus(BaseModel):
    progress: float
    best_parameters: Dict[str, Any]

class ABStatus(BaseModel):
    test_name: str
    variant_a_winrate: float
    variant_b_winrate: float
    winner: str

# --- Endpoints --- #

@app.get("/pnl/summary", response_model=PnLSummary)
def get_pnl_summary():
    return PnLSummary(
        total_trades=random.randint(50, 500),
        win_rate=round(random.uniform(45, 75), 2),
        profit_factor=round(random.uniform(1.1, 3.0), 2),
        net_profit=round(random.uniform(-500, 2500), 2),
    )

@app.get("/autotune/status", response_model=AutoTuneStatus)
def get_autotune_status():
    params = {"rsi_length": random.randint(5, 20), "ema_length": random.randint(10, 50)}
    return AutoTuneStatus(progress=round(random.uniform(0, 100), 2), best_parameters=params)

@app.get("/ab/status", response_model=ABStatus)
def get_ab_status():
    a_win = round(random.uniform(40, 60), 2)
    b_win = round(random.uniform(40, 60), 2)
    winner = "A" if a_win > b_win else "B"
    return ABStatus(
        test_name="RSI vs EMA",
        variant_a_winrate=a_win,
        variant_b_winrate=b_win,
        winner=winner,
    )

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# Run with: uvicorn app:app --reload --port 8080
