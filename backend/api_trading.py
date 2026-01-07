# backend/api_trading.py
"""
API endpoints for autonomous trading system.

Provides:
- Trading settings management
- Signal queue operations
- Trade execution
- Kill switch controls
- Position and P&L monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

from backend.autonomous_trading import (
    TradingSettingsManager,
    TradingSettings,
    TradingMode,
    RiskProfile,
    PendingSignalQueue,
    PendingSignal,
    SignalStatus,
    DailyStatsManager,
    TradingActivityLogger,
    RiskManager,
    AutonomousTradingEngine,
    init_trading_schema
)

# Initialize schema
try:
    init_trading_schema()
except Exception as e:
    print(f"[WARN] Trading schema init: {e}")

router = APIRouter(prefix="/trading", tags=["Autonomous Trading"])

# Global trading engine instance
_trading_engines: Dict[int, AutonomousTradingEngine] = {}


def get_trading_engine(user_id: int = 1) -> AutonomousTradingEngine:
    """Get or create trading engine for user."""
    if user_id not in _trading_engines:
        _trading_engines[user_id] = AutonomousTradingEngine(user_id)
    return _trading_engines[user_id]


# ============================================================================
# Request/Response Models
# ============================================================================

class TradingSettingsUpdate(BaseModel):
    """Update trading settings request"""
    trading_mode: Optional[str] = None
    risk_profile: Optional[str] = None
    min_confidence_auto: Optional[float] = Field(None, ge=0, le=1)
    min_confidence_manual: Optional[float] = Field(None, ge=0, le=1)
    min_consensus: Optional[float] = Field(None, ge=0, le=1)
    max_position_size_usd: Optional[float] = Field(None, ge=0)
    max_portfolio_pct: Optional[float] = Field(None, ge=0, le=1)
    max_open_positions: Optional[int] = Field(None, ge=1)
    max_trades_per_day: Optional[int] = Field(None, ge=1)
    max_daily_loss_usd: Optional[float] = Field(None, ge=0)
    allowed_symbols: Optional[List[str]] = None
    blocked_symbols: Optional[List[str]] = None
    signal_expiry_minutes: Optional[int] = Field(None, ge=1)
    cooldown_minutes: Optional[int] = Field(None, ge=0)
    paper_trading: Optional[bool] = None


class TradingSettingsResponse(BaseModel):
    """Trading settings response"""
    trading_mode: str
    risk_profile: str
    min_confidence_auto: float
    min_confidence_manual: float
    min_consensus: float
    max_position_size_usd: float
    max_portfolio_pct: float
    max_open_positions: int
    max_trades_per_day: int
    max_daily_loss_usd: float
    allowed_symbols: List[str]
    blocked_symbols: List[str]
    signal_expiry_minutes: int
    cooldown_minutes: int
    kill_switch_active: bool
    kill_switch_reason: str
    paper_trading: bool


class KillSwitchRequest(BaseModel):
    """Kill switch activation request"""
    reason: str = "Manual activation"


class SignalApprovalRequest(BaseModel):
    """Signal approval/rejection request"""
    signal_id: int
    action: str = Field(..., pattern="^(approve|reject)$")
    reason: Optional[str] = None


class ManualSignalRequest(BaseModel):
    """Manual signal submission"""
    symbol: str
    action: str = Field(..., pattern="^(BUY|SELL)$")
    confidence: float = Field(0.8, ge=0, le=1)
    size_usd: Optional[float] = None
    reason: str = "Manual signal"


class SignalResponse(BaseModel):
    """Pending signal response"""
    id: int
    signal_source: str
    symbol: str
    action: str
    confidence: float
    consensus: float
    recommended_size_usd: float
    status: str
    status_reason: str
    current_price: float
    created_at: str
    expires_at: Optional[str]
    actioned_at: Optional[str]
    order_id: Optional[int]
    executed_price: Optional[float]


class DailyStatsResponse(BaseModel):
    """Daily trading stats response"""
    date: str
    trades_count: int
    realized_pnl: float
    unrealized_pnl: float
    wins: int
    losses: int
    volume_usd: float
    win_rate: float


class TradingStatusResponse(BaseModel):
    """Overall trading status response"""
    trading_enabled: bool
    trading_mode: str
    kill_switch_active: bool
    paper_trading: bool
    pending_signals_count: int
    today_trades: int
    today_pnl: float
    daily_limit_remaining: int
    loss_limit_remaining: float


# ============================================================================
# Settings Endpoints
# ============================================================================

@router.get("/settings", response_model=TradingSettingsResponse)
async def get_trading_settings(user_id: int = 1):
    """Get current trading settings."""
    settings = TradingSettingsManager.get_settings(user_id)
    return TradingSettingsResponse(
        trading_mode=settings.trading_mode.value,
        risk_profile=settings.risk_profile.value,
        min_confidence_auto=settings.min_confidence_auto,
        min_confidence_manual=settings.min_confidence_manual,
        min_consensus=settings.min_consensus,
        max_position_size_usd=settings.max_position_size_usd,
        max_portfolio_pct=settings.max_portfolio_pct,
        max_open_positions=settings.max_open_positions,
        max_trades_per_day=settings.max_trades_per_day,
        max_daily_loss_usd=settings.max_daily_loss_usd,
        allowed_symbols=settings.allowed_symbols,
        blocked_symbols=settings.blocked_symbols,
        signal_expiry_minutes=settings.signal_expiry_minutes,
        cooldown_minutes=settings.cooldown_minutes,
        kill_switch_active=settings.kill_switch_active,
        kill_switch_reason=settings.kill_switch_reason,
        paper_trading=settings.paper_trading
    )


@router.put("/settings", response_model=TradingSettingsResponse)
async def update_trading_settings(updates: TradingSettingsUpdate, user_id: int = 1):
    """Update trading settings."""
    update_dict = {k: v for k, v in updates.dict().items() if v is not None}
    settings = TradingSettingsManager.update_settings(user_id, update_dict)

    return TradingSettingsResponse(
        trading_mode=settings.trading_mode.value,
        risk_profile=settings.risk_profile.value,
        min_confidence_auto=settings.min_confidence_auto,
        min_confidence_manual=settings.min_confidence_manual,
        min_consensus=settings.min_consensus,
        max_position_size_usd=settings.max_position_size_usd,
        max_portfolio_pct=settings.max_portfolio_pct,
        max_open_positions=settings.max_open_positions,
        max_trades_per_day=settings.max_trades_per_day,
        max_daily_loss_usd=settings.max_daily_loss_usd,
        allowed_symbols=settings.allowed_symbols,
        blocked_symbols=settings.blocked_symbols,
        signal_expiry_minutes=settings.signal_expiry_minutes,
        cooldown_minutes=settings.cooldown_minutes,
        kill_switch_active=settings.kill_switch_active,
        kill_switch_reason=settings.kill_switch_reason,
        paper_trading=settings.paper_trading
    )


# ============================================================================
# Kill Switch Endpoints
# ============================================================================

@router.post("/kill-switch/activate")
async def activate_kill_switch(request: KillSwitchRequest, user_id: int = 1):
    """Activate emergency kill switch - stops all trading immediately."""
    settings = TradingSettingsManager.activate_kill_switch(user_id, request.reason)
    return {
        "success": True,
        "message": "Kill switch activated",
        "kill_switch_active": settings.kill_switch_active,
        "reason": settings.kill_switch_reason,
        "timestamp": settings.kill_switch_timestamp
    }


@router.post("/kill-switch/deactivate")
async def deactivate_kill_switch(user_id: int = 1):
    """Deactivate kill switch - resume trading."""
    settings = TradingSettingsManager.deactivate_kill_switch(user_id)
    return {
        "success": True,
        "message": "Kill switch deactivated",
        "kill_switch_active": settings.kill_switch_active
    }


@router.get("/kill-switch/status")
async def get_kill_switch_status(user_id: int = 1):
    """Get current kill switch status."""
    settings = TradingSettingsManager.get_settings(user_id)
    return {
        "kill_switch_active": settings.kill_switch_active,
        "reason": settings.kill_switch_reason,
        "timestamp": settings.kill_switch_timestamp
    }


# ============================================================================
# Signal Queue Endpoints
# ============================================================================

@router.get("/signals/pending", response_model=List[SignalResponse])
async def get_pending_signals(user_id: int = 1):
    """Get all pending signals awaiting action."""
    signals = PendingSignalQueue.get_pending(user_id)
    return [
        SignalResponse(
            id=s.id,
            signal_source=s.signal_source,
            symbol=s.symbol,
            action=s.action,
            confidence=s.confidence,
            consensus=s.consensus,
            recommended_size_usd=s.recommended_size_usd or 0,
            status=s.status.value if isinstance(s.status, SignalStatus) else s.status,
            status_reason=s.status_reason,
            current_price=s.current_price or 0,
            created_at=s.created_at,
            expires_at=s.expires_at,
            actioned_at=s.actioned_at,
            order_id=s.order_id,
            executed_price=s.executed_price
        )
        for s in signals
    ]


@router.get("/signals/history", response_model=List[SignalResponse])
async def get_signal_history(user_id: int = 1, limit: int = 100):
    """Get historical signals (executed, rejected, expired)."""
    signals = PendingSignalQueue.get_history(user_id, limit)
    return [
        SignalResponse(
            id=s.id,
            signal_source=s.signal_source,
            symbol=s.symbol,
            action=s.action,
            confidence=s.confidence,
            consensus=s.consensus,
            recommended_size_usd=s.recommended_size_usd or 0,
            status=s.status.value if isinstance(s.status, SignalStatus) else s.status,
            status_reason=s.status_reason,
            current_price=s.current_price or 0,
            created_at=s.created_at,
            expires_at=s.expires_at,
            actioned_at=s.actioned_at,
            order_id=s.order_id,
            executed_price=s.executed_price
        )
        for s in signals
    ]


@router.get("/signals/{signal_id}")
async def get_signal(signal_id: int):
    """Get details of a specific signal."""
    signal = PendingSignalQueue.get_signal(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    return {
        "id": signal.id,
        "signal_source": signal.signal_source,
        "symbol": signal.symbol,
        "action": signal.action,
        "confidence": signal.confidence,
        "consensus": signal.consensus,
        "recommended_size_usd": signal.recommended_size_usd,
        "recommended_size_pct": signal.recommended_size_pct,
        "signal_data": signal.signal_data,
        "aggregation_data": signal.aggregation_data,
        "current_price": signal.current_price,
        "market_context": signal.market_context,
        "status": signal.status.value if isinstance(signal.status, SignalStatus) else signal.status,
        "status_reason": signal.status_reason,
        "order_id": signal.order_id,
        "executed_price": signal.executed_price,
        "executed_qty": signal.executed_qty,
        "execution_error": signal.execution_error,
        "created_at": signal.created_at,
        "expires_at": signal.expires_at,
        "actioned_at": signal.actioned_at
    }


@router.post("/signals/{signal_id}/approve")
async def approve_signal(signal_id: int, user_id: int = 1):
    """Approve and execute a pending signal."""
    engine = get_trading_engine(user_id)
    result = await engine.approve_signal(signal_id)

    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))

    return result


@router.post("/signals/{signal_id}/reject")
async def reject_signal(signal_id: int, reason: str = "Manual rejection", user_id: int = 1):
    """Reject a pending signal."""
    engine = get_trading_engine(user_id)
    result = await engine.reject_signal(signal_id, reason)

    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))

    return result


@router.post("/signals/batch-action")
async def batch_signal_action(request: SignalApprovalRequest, user_id: int = 1):
    """Approve or reject a signal."""
    engine = get_trading_engine(user_id)

    if request.action == "approve":
        result = await engine.approve_signal(request.signal_id)
    else:
        result = await engine.reject_signal(request.signal_id, request.reason or "Manual rejection")

    return result


# ============================================================================
# Manual Trading Endpoints
# ============================================================================

@router.post("/signals/manual")
async def submit_manual_signal(request: ManualSignalRequest, user_id: int = 1):
    """Submit a manual trading signal."""
    settings = TradingSettingsManager.get_settings(user_id)

    # Calculate position size if not provided
    if request.size_usd:
        size_usd = request.size_usd
    else:
        sizing = RiskManager.calculate_position_size(request.confidence, settings)
        size_usd = sizing['size_usd']

    # Create signal
    signal = PendingSignal(
        signal_source="manual",
        symbol=request.symbol,
        action=request.action,
        confidence=request.confidence,
        consensus=1.0,  # Manual signals are 100% consensus
        recommended_size_usd=size_usd,
        signal_data={"reason": request.reason},
        status=SignalStatus.PENDING
    )

    signal_id = PendingSignalQueue.add_signal(signal)

    TradingActivityLogger.log(user_id, "manual_signal_submitted", {
        "signal_id": signal_id,
        "symbol": request.symbol,
        "action": request.action
    })

    return {
        "success": True,
        "signal_id": signal_id,
        "message": f"Manual {request.action} signal created for {request.symbol}"
    }


@router.post("/execute/{signal_id}")
async def execute_signal(signal_id: int, user_id: int = 1):
    """Directly execute a signal (bypasses approval queue)."""
    engine = get_trading_engine(user_id)

    # First approve it
    signal = PendingSignalQueue.get_signal(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    PendingSignalQueue.update_status(signal_id, SignalStatus.APPROVED, reason="Direct execution")
    result = await engine.execute_signal(signal_id)

    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))

    return result


# ============================================================================
# Status & Stats Endpoints
# ============================================================================

@router.get("/status", response_model=TradingStatusResponse)
async def get_trading_status(user_id: int = 1):
    """Get overall trading system status."""
    settings = TradingSettingsManager.get_settings(user_id)
    daily_stats = DailyStatsManager.get_today(user_id)
    pending_signals = PendingSignalQueue.get_pending(user_id)

    return TradingStatusResponse(
        trading_enabled=settings.trading_mode != TradingMode.OFF,
        trading_mode=settings.trading_mode.value,
        kill_switch_active=settings.kill_switch_active,
        paper_trading=settings.paper_trading,
        pending_signals_count=len(pending_signals),
        today_trades=daily_stats.trades_count,
        today_pnl=daily_stats.realized_pnl,
        daily_limit_remaining=max(0, settings.max_trades_per_day - daily_stats.trades_count),
        loss_limit_remaining=max(0, settings.max_daily_loss_usd + daily_stats.realized_pnl)
    )


@router.get("/stats/daily", response_model=DailyStatsResponse)
async def get_daily_stats(user_id: int = 1):
    """Get today's trading statistics."""
    stats = DailyStatsManager.get_today(user_id)
    win_rate = (stats.wins / stats.trades_count * 100) if stats.trades_count > 0 else 0

    return DailyStatsResponse(
        date=stats.date,
        trades_count=stats.trades_count,
        realized_pnl=stats.realized_pnl,
        unrealized_pnl=stats.unrealized_pnl,
        wins=stats.wins,
        losses=stats.losses,
        volume_usd=stats.volume_usd,
        win_rate=round(win_rate, 2)
    )


@router.get("/activity")
async def get_activity_log(user_id: int = 1, limit: int = 50):
    """Get recent trading activity log."""
    activities = TradingActivityLogger.get_recent(user_id, limit)
    return {"activities": activities}


# ============================================================================
# Risk Check Endpoint
# ============================================================================

@router.get("/risk-check/{symbol}")
async def check_trading_risk(symbol: str, size_usd: float = 100, user_id: int = 1):
    """Check if a trade would be allowed given current risk limits."""
    settings = TradingSettingsManager.get_settings(user_id)
    result = RiskManager.check_can_trade(user_id, symbol, size_usd, settings)

    return {
        "symbol": symbol,
        "size_usd": size_usd,
        "allowed": result['allowed'],
        "reasons": result['reasons'],
        "current_settings": {
            "trading_mode": settings.trading_mode.value,
            "kill_switch_active": settings.kill_switch_active,
            "max_position_size_usd": settings.max_position_size_usd,
            "daily_loss_limit": settings.max_daily_loss_usd
        }
    }


# ============================================================================
# Position Sizing Endpoint
# ============================================================================

@router.get("/position-size")
async def calculate_position_size(
    confidence: float = 0.7,
    account_balance: float = None,
    user_id: int = 1
):
    """Calculate recommended position size for given confidence level."""
    settings = TradingSettingsManager.get_settings(user_id)
    sizing = RiskManager.calculate_position_size(confidence, settings, account_balance)

    return {
        "confidence": confidence,
        "risk_profile": settings.risk_profile.value,
        "recommended_size_usd": sizing['size_usd'],
        "recommended_size_pct": sizing['size_pct'],
        "confidence_factor": sizing['confidence_factor'],
        "risk_multiplier": sizing['risk_multiplier'],
        "max_position_size": settings.max_position_size_usd
    }


# ============================================================================
# Autonomous Loop Control
# ============================================================================

@router.post("/autonomous/start")
async def start_autonomous_loop(
    background_tasks: BackgroundTasks,
    interval_seconds: int = 30,
    user_id: int = 1
):
    """Start the autonomous trading loop."""
    engine = get_trading_engine(user_id)

    if engine._running:
        return {"success": False, "message": "Autonomous loop already running"}

    background_tasks.add_task(engine.run_autonomous_loop, interval_seconds)

    return {
        "success": True,
        "message": f"Autonomous trading loop started with {interval_seconds}s interval"
    }


@router.post("/autonomous/stop")
async def stop_autonomous_loop(user_id: int = 1):
    """Stop the autonomous trading loop."""
    engine = get_trading_engine(user_id)
    engine.stop_loop()

    return {"success": True, "message": "Autonomous trading loop stopped"}


@router.get("/autonomous/status")
async def get_autonomous_status(user_id: int = 1):
    """Get autonomous loop status."""
    engine = get_trading_engine(user_id)

    return {
        "running": engine._running,
        "user_id": user_id
    }


# ============================================================================
# Exchange Management Endpoints
# ============================================================================

try:
    from backend.brokers.exchange_config import (
        get_all_exchanges,
        get_configured_exchanges,
        get_exchange_config,
        COMMON_PAIRS
    )
    EXCHANGE_CONFIG_AVAILABLE = True
except ImportError:
    EXCHANGE_CONFIG_AVAILABLE = False


@router.get("/exchanges")
async def list_exchanges():
    """
    List all supported exchanges.

    Returns exchange information including:
    - Supported features (spot, futures, margin)
    - Fee structure
    - Testnet availability
    """
    if not EXCHANGE_CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Exchange config module not available")

    return {
        "exchanges": get_all_exchanges(),
        "configured": get_configured_exchanges(),
        "common_pairs": COMMON_PAIRS
    }


@router.get("/exchanges/{exchange_id}")
async def get_exchange_info(exchange_id: str):
    """Get detailed information about a specific exchange."""
    if not EXCHANGE_CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Exchange config module not available")

    config = get_exchange_config(exchange_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Exchange '{exchange_id}' not found")

    # Check if configured
    configured_exchanges = get_configured_exchanges()
    is_configured = exchange_id.lower() in configured_exchanges

    return {
        "id": config.exchange_id,
        "name": config.display_name,
        "configured": is_configured,
        "features": {
            "spot": config.has_spot,
            "futures": config.has_futures,
            "margin": config.has_margin,
        },
        "fees": {
            "maker": config.maker_fee,
            "taker": config.taker_fee,
        },
        "testnet_available": config.testnet_available,
        "requires_password": config.requires_password,
        "supported_quotes": config.supported_quote_currencies,
        "min_order_value_usd": config.min_order_value_usd,
        "notes": config.notes
    }
