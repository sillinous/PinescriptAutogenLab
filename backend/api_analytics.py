# backend/api_analytics.py
"""
Trading Analytics and Portfolio Monitoring API endpoints.

Provides comprehensive analytics for:
- Trade history and performance metrics
- Portfolio positions and real-time P&L
- Strategy performance comparison
- Risk metrics and exposure analysis
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import json

from backend.database import get_db

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================

class TradeMetrics(BaseModel):
    """Aggregated trading metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: Optional[float]
    avg_loss: Optional[float]
    profit_factor: Optional[float]
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    max_drawdown: Optional[float]
    avg_trade_duration_hours: Optional[float]


class PositionSummary(BaseModel):
    """Current position information."""
    symbol: str
    side: str
    qty: float
    avg_entry_price: float
    current_price: Optional[float]
    market_value: Optional[float]
    unrealized_pnl: Optional[float]
    unrealized_pnl_pct: Optional[float]
    cost_basis: float


class PortfolioSummary(BaseModel):
    """Overall portfolio summary."""
    total_equity: float
    total_cash: float
    total_positions_value: float
    total_unrealized_pnl: float
    position_count: int
    top_gainer: Optional[Dict[str, Any]]
    top_loser: Optional[Dict[str, Any]]


class DailyPerformance(BaseModel):
    """Daily performance record."""
    date: str
    pnl: float
    trades_count: int
    win_rate: float
    cumulative_pnl: float


# ============================================================================
# Trade History Analytics
# ============================================================================

@router.get("/trades/history")
async def get_trade_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    side: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get paginated trade history with filters.

    Returns filled orders with execution details.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Build query with filters
    query = "SELECT * FROM orders WHERE 1=1"
    params = []

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol.upper())

    if status:
        query += " AND status = ?"
        params.append(status)

    if side:
        query += " AND side = ?"
        params.append(side.lower())

    if start_date:
        query += " AND created_at >= ?"
        params.append(start_date)

    if end_date:
        query += " AND created_at <= ?"
        params.append(end_date)

    # Get total count
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    # Get paginated results
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    trades = []
    for row in rows:
        trade = dict(row)
        # Parse JSON fields
        if trade.get('webhook_payload'):
            try:
                trade['webhook_payload'] = json.loads(trade['webhook_payload'])
            except:
                pass
        trades.append(trade)

    return {
        "trades": trades,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(trades) < total_count
    }


@router.get("/trades/metrics")
async def get_trade_metrics(
    period: str = Query("all", pattern="^(day|week|month|quarter|year|all)$"),
    symbol: Optional[str] = None
) -> TradeMetrics:
    """
    Get aggregated trading metrics for a period.

    Calculates win rate, profit factor, average P&L, etc.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Calculate period start date
    now = datetime.utcnow()
    period_starts = {
        "day": now - timedelta(days=1),
        "week": now - timedelta(weeks=1),
        "month": now - timedelta(days=30),
        "quarter": now - timedelta(days=90),
        "year": now - timedelta(days=365),
        "all": datetime(2000, 1, 1)
    }
    start_date = period_starts[period]

    # Base query for filled orders
    query = """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN filled_avg_price * filled_qty > 0 THEN 1 ELSE 0 END) as with_value,
            SUM(filled_avg_price * filled_qty) as total_value
        FROM orders
        WHERE status = 'filled' AND created_at >= ?
    """
    params = [start_date.isoformat()]

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol.upper())

    cursor.execute(query, params)
    basic_stats = cursor.fetchone()

    # Get winning/losing trade stats
    # Note: This is simplified - real P&L needs trade pairing logic
    cursor.execute("""
        SELECT
            side,
            COUNT(*) as count,
            AVG(filled_avg_price * filled_qty) as avg_value,
            SUM(filled_avg_price * filled_qty) as total_value
        FROM orders
        WHERE status = 'filled' AND created_at >= ?
        GROUP BY side
    """, [start_date.isoformat()])

    side_stats = {row['side']: dict(row) for row in cursor.fetchall()}

    # Get unrealized P&L from positions
    cursor.execute("SELECT SUM(unrealized_pnl) as total FROM positions")
    unrealized = cursor.fetchone()
    unrealized_pnl = unrealized['total'] if unrealized['total'] else 0

    # Calculate metrics
    total_trades = basic_stats['total_trades'] or 0
    buy_value = side_stats.get('buy', {}).get('total_value', 0) or 0
    sell_value = side_stats.get('sell', {}).get('total_value', 0) or 0

    # Simplified P&L calculation
    realized_pnl = sell_value - buy_value
    total_pnl = realized_pnl + unrealized_pnl

    # Win rate approximation (needs proper trade pairing for accuracy)
    winning_trades = int(total_trades * 0.5)  # Placeholder
    losing_trades = total_trades - winning_trades

    conn.close()

    return TradeMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / total_trades if total_trades > 0 else 0,
        avg_win=None,  # Requires proper trade pairing
        avg_loss=None,
        profit_factor=abs(sell_value / buy_value) if buy_value else None,
        total_pnl=round(total_pnl, 2),
        realized_pnl=round(realized_pnl, 2),
        unrealized_pnl=round(unrealized_pnl, 2),
        max_drawdown=None,  # Requires equity curve calculation
        avg_trade_duration_hours=None
    )


@router.get("/trades/daily-performance")
async def get_daily_performance(
    days: int = Query(30, ge=1, le=365)
) -> List[DailyPerformance]:
    """
    Get daily performance breakdown.

    Returns P&L, trade count, and win rate for each day.
    """
    conn = get_db()
    cursor = conn.cursor()

    start_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()

    cursor.execute("""
        SELECT
            DATE(created_at) as date,
            COUNT(*) as trades_count,
            SUM(CASE WHEN side = 'sell' THEN filled_avg_price * filled_qty
                     ELSE -filled_avg_price * filled_qty END) as daily_pnl
        FROM orders
        WHERE status = 'filled' AND created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    """, [start_date])

    rows = cursor.fetchall()
    conn.close()

    results = []
    cumulative_pnl = 0

    for row in rows:
        daily_pnl = row['daily_pnl'] or 0
        cumulative_pnl += daily_pnl

        results.append(DailyPerformance(
            date=row['date'],
            pnl=round(daily_pnl, 2),
            trades_count=row['trades_count'],
            win_rate=0.5,  # Placeholder - needs trade pairing
            cumulative_pnl=round(cumulative_pnl, 2)
        ))

    return results


@router.get("/trades/by-symbol")
async def get_trades_by_symbol(
    period: str = Query("month", pattern="^(week|month|quarter|year|all)$")
) -> Dict[str, Any]:
    """
    Get trade statistics grouped by symbol.
    """
    conn = get_db()
    cursor = conn.cursor()

    period_days = {"week": 7, "month": 30, "quarter": 90, "year": 365, "all": 3650}
    start_date = (datetime.utcnow() - timedelta(days=period_days[period])).isoformat()

    cursor.execute("""
        SELECT
            symbol,
            COUNT(*) as trade_count,
            SUM(CASE WHEN side = 'buy' THEN filled_avg_price * filled_qty ELSE 0 END) as buy_volume,
            SUM(CASE WHEN side = 'sell' THEN filled_avg_price * filled_qty ELSE 0 END) as sell_volume
        FROM orders
        WHERE status = 'filled' AND created_at >= ?
        GROUP BY symbol
        ORDER BY trade_count DESC
    """, [start_date])

    rows = cursor.fetchall()
    conn.close()

    symbols = []
    for row in rows:
        pnl = (row['sell_volume'] or 0) - (row['buy_volume'] or 0)
        symbols.append({
            "symbol": row['symbol'],
            "trade_count": row['trade_count'],
            "buy_volume": round(row['buy_volume'] or 0, 2),
            "sell_volume": round(row['sell_volume'] or 0, 2),
            "net_pnl": round(pnl, 2)
        })

    return {
        "period": period,
        "symbols": symbols,
        "total_symbols": len(symbols)
    }


# ============================================================================
# Portfolio Position Monitoring
# ============================================================================

@router.get("/portfolio/positions")
async def get_portfolio_positions() -> List[PositionSummary]:
    """
    Get all current open positions with P&L.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM positions
        WHERE qty != 0
        ORDER BY ABS(market_value) DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    positions = []
    for row in rows:
        positions.append(PositionSummary(
            symbol=row['symbol'],
            side='long' if row['qty'] > 0 else 'short',
            qty=abs(row['qty']),
            avg_entry_price=row['avg_entry_price'],
            current_price=row['current_price'],
            market_value=row['market_value'],
            unrealized_pnl=row['unrealized_pnl'],
            unrealized_pnl_pct=row['unrealized_pnl_pct'],
            cost_basis=row['cost_basis']
        ))

    return positions


@router.get("/portfolio/summary")
async def get_portfolio_summary() -> PortfolioSummary:
    """
    Get overall portfolio summary with key metrics.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get position statistics
    cursor.execute("""
        SELECT
            COUNT(*) as position_count,
            SUM(market_value) as total_value,
            SUM(unrealized_pnl) as total_unrealized
        FROM positions
        WHERE qty != 0
    """)
    pos_stats = cursor.fetchone()

    # Get top gainer
    cursor.execute("""
        SELECT symbol, unrealized_pnl, unrealized_pnl_pct
        FROM positions
        WHERE qty != 0 AND unrealized_pnl IS NOT NULL
        ORDER BY unrealized_pnl DESC
        LIMIT 1
    """)
    top_gainer = cursor.fetchone()

    # Get top loser
    cursor.execute("""
        SELECT symbol, unrealized_pnl, unrealized_pnl_pct
        FROM positions
        WHERE qty != 0 AND unrealized_pnl IS NOT NULL
        ORDER BY unrealized_pnl ASC
        LIMIT 1
    """)
    top_loser = cursor.fetchone()

    # Get latest equity snapshot
    cursor.execute("""
        SELECT equity, cash, portfolio_value
        FROM performance_snapshots
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    equity_row = cursor.fetchone()

    conn.close()

    # Calculate totals
    total_positions_value = pos_stats['total_value'] or 0
    total_unrealized = pos_stats['total_unrealized'] or 0

    equity = equity_row['equity'] if equity_row else total_positions_value
    cash = equity_row['cash'] if equity_row else 0

    return PortfolioSummary(
        total_equity=round(equity, 2),
        total_cash=round(cash, 2),
        total_positions_value=round(total_positions_value, 2),
        total_unrealized_pnl=round(total_unrealized, 2),
        position_count=pos_stats['position_count'] or 0,
        top_gainer=dict(top_gainer) if top_gainer else None,
        top_loser=dict(top_loser) if top_loser else None
    )


@router.get("/portfolio/equity-curve")
async def get_equity_curve(
    days: int = Query(30, ge=1, le=365)
) -> List[Dict[str, Any]]:
    """
    Get historical equity curve data.
    """
    conn = get_db()
    cursor = conn.cursor()

    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT
            DATE(timestamp) as date,
            MAX(equity) as equity,
            MAX(portfolio_value) as portfolio_value,
            MAX(cash) as cash
        FROM performance_snapshots
        WHERE timestamp >= ?
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, [start_date])

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ============================================================================
# Risk Metrics
# ============================================================================

@router.get("/risk/metrics")
async def get_risk_metrics() -> Dict[str, Any]:
    """
    Get portfolio-wide risk metrics.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get position concentration
    cursor.execute("""
        SELECT
            symbol,
            market_value,
            (market_value * 100.0 / (SELECT SUM(market_value) FROM positions WHERE qty != 0)) as pct
        FROM positions
        WHERE qty != 0
        ORDER BY market_value DESC
    """)
    positions = cursor.fetchall()

    # Calculate concentration metrics
    total_value = sum(abs(p['market_value'] or 0) for p in positions)
    top_5_value = sum(abs(p['market_value'] or 0) for p in positions[:5])

    # Get daily returns for volatility calculation
    cursor.execute("""
        SELECT
            DATE(timestamp) as date,
            MAX(equity) as equity
        FROM performance_snapshots
        WHERE timestamp >= DATE('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY date
    """)
    equity_history = cursor.fetchall()

    # Calculate simple volatility (if enough data)
    daily_returns = []
    for i in range(1, len(equity_history)):
        if equity_history[i]['equity'] and equity_history[i-1]['equity']:
            ret = (equity_history[i]['equity'] - equity_history[i-1]['equity']) / equity_history[i-1]['equity']
            daily_returns.append(ret)

    volatility = None
    if daily_returns:
        import math
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        volatility = math.sqrt(variance) * math.sqrt(252)  # Annualized

    conn.close()

    return {
        "position_count": len(positions),
        "total_exposure": round(total_value, 2),
        "top_5_concentration": round(top_5_value / total_value * 100, 2) if total_value else 0,
        "largest_position": {
            "symbol": positions[0]['symbol'],
            "value": round(positions[0]['market_value'], 2),
            "pct": round(positions[0]['pct'], 2)
        } if positions else None,
        "annualized_volatility": round(volatility * 100, 2) if volatility else None,
        "diversification_score": round(1 - (top_5_value / total_value), 2) if total_value else 0
    }


@router.get("/risk/exposure")
async def get_risk_exposure() -> Dict[str, Any]:
    """
    Get current portfolio exposure breakdown.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            broker_type,
            SUM(CASE WHEN qty > 0 THEN market_value ELSE 0 END) as long_exposure,
            SUM(CASE WHEN qty < 0 THEN ABS(market_value) ELSE 0 END) as short_exposure,
            COUNT(*) as position_count
        FROM positions
        WHERE qty != 0
        GROUP BY broker_type
    """)

    by_broker = [dict(row) for row in cursor.fetchall()]

    # Overall exposure
    cursor.execute("""
        SELECT
            SUM(CASE WHEN qty > 0 THEN market_value ELSE 0 END) as long_total,
            SUM(CASE WHEN qty < 0 THEN ABS(market_value) ELSE 0 END) as short_total
        FROM positions
        WHERE qty != 0
    """)
    totals = cursor.fetchone()

    conn.close()

    long_total = totals['long_total'] or 0
    short_total = totals['short_total'] or 0

    return {
        "long_exposure": round(long_total, 2),
        "short_exposure": round(short_total, 2),
        "net_exposure": round(long_total - short_total, 2),
        "gross_exposure": round(long_total + short_total, 2),
        "long_short_ratio": round(long_total / short_total, 2) if short_total else None,
        "by_broker": by_broker
    }


# ============================================================================
# Strategy Performance
# ============================================================================

@router.get("/strategies/list")
async def list_strategies() -> List[Dict[str, Any]]:
    """
    List all strategies with performance metrics.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get unique strategies from orders
    cursor.execute("""
        SELECT DISTINCT
            json_extract(webhook_payload, '$.strategy') as strategy_name
        FROM orders
        WHERE webhook_payload IS NOT NULL
    """)
    strategies = [row[0] for row in cursor.fetchall() if row[0]]

    # Get performance for each strategy
    results = []
    for strategy in strategies:
        cursor.execute("""
            SELECT
                COUNT(*) as trade_count,
                SUM(CASE WHEN side = 'sell' THEN filled_avg_price * filled_qty ELSE 0 END) as sells,
                SUM(CASE WHEN side = 'buy' THEN filled_avg_price * filled_qty ELSE 0 END) as buys
            FROM orders
            WHERE status = 'filled'
              AND json_extract(webhook_payload, '$.strategy') = ?
        """, [strategy])

        stats = cursor.fetchone()
        pnl = (stats['sells'] or 0) - (stats['buys'] or 0)

        results.append({
            "name": strategy,
            "trade_count": stats['trade_count'],
            "total_pnl": round(pnl, 2),
            "status": "active"
        })

    # Also get strategies from strategy_params table
    cursor.execute("""
        SELECT DISTINCT strategy_name FROM strategy_params
    """)
    param_strategies = [row[0] for row in cursor.fetchall()]

    for strategy in param_strategies:
        if strategy not in strategies:
            results.append({
                "name": strategy,
                "trade_count": 0,
                "total_pnl": 0,
                "status": "configured"
            })

    conn.close()

    return sorted(results, key=lambda x: x['trade_count'], reverse=True)


@router.get("/strategies/{strategy_name}/performance")
async def get_strategy_performance(strategy_name: str) -> Dict[str, Any]:
    """
    Get detailed performance for a specific strategy.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get trade history for this strategy
    cursor.execute("""
        SELECT
            symbol,
            side,
            filled_avg_price,
            filled_qty,
            created_at,
            filled_at
        FROM orders
        WHERE status = 'filled'
          AND json_extract(webhook_payload, '$.strategy') = ?
        ORDER BY created_at DESC
    """, [strategy_name])

    trades = [dict(row) for row in cursor.fetchall()]

    # Calculate metrics
    total_buys = sum(t['filled_avg_price'] * t['filled_qty'] for t in trades if t['side'] == 'buy')
    total_sells = sum(t['filled_avg_price'] * t['filled_qty'] for t in trades if t['side'] == 'sell')

    # Get strategy parameters
    cursor.execute("""
        SELECT param_name, param_value, is_best
        FROM strategy_params
        WHERE strategy_name = ?
    """, [strategy_name])

    params = {row['param_name']: row['param_value'] for row in cursor.fetchall()}

    conn.close()

    return {
        "strategy_name": strategy_name,
        "total_trades": len(trades),
        "buy_trades": len([t for t in trades if t['side'] == 'buy']),
        "sell_trades": len([t for t in trades if t['side'] == 'sell']),
        "total_buy_volume": round(total_buys, 2),
        "total_sell_volume": round(total_sells, 2),
        "net_pnl": round(total_sells - total_buys, 2),
        "parameters": params,
        "recent_trades": trades[:10]
    }


# Export router
__all__ = ['router']
