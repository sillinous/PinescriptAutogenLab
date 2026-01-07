# backend/autonomous_trading.py
"""
Autonomous Trading System for PineLab

This module provides the core infrastructure for autonomous trading with user controls:
- User trading settings management
- Signal-to-execution pipeline
- Risk management and position sizing
- Autonomous trading loop
- Kill switch and emergency controls
- Pending orders queue for manual approval
"""

import asyncio
import sqlite3
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import os

# Database path
DATA_DIR = Path(os.getenv("PINELAB_DATA", "./data"))
DB_PATH = DATA_DIR / "pinelab.db"


class RiskProfile(str, Enum):
    """User risk tolerance levels"""
    CONSERVATIVE = "conservative"  # Lower position sizes, higher confidence required
    MODERATE = "moderate"          # Balanced approach
    AGGRESSIVE = "aggressive"      # Larger positions, lower confidence threshold


class TradingMode(str, Enum):
    """Trading operation modes"""
    OFF = "off"                    # No trading at all
    MANUAL = "manual"              # All signals require manual approval
    SEMI_AUTO = "semi_auto"        # High-confidence auto, rest manual
    FULL_AUTO = "full_auto"        # Fully autonomous


class SignalStatus(str, Enum):
    """Status of trading signals in the queue"""
    PENDING = "pending"            # Awaiting decision
    APPROVED = "approved"          # User approved, ready to execute
    AUTO_APPROVED = "auto_approved"  # System auto-approved
    EXECUTING = "executing"        # Currently being executed
    EXECUTED = "executed"          # Successfully executed
    REJECTED = "rejected"          # User rejected
    EXPIRED = "expired"            # Signal expired before action
    FAILED = "failed"              # Execution failed


@dataclass
class TradingSettings:
    """User trading configuration"""
    user_id: int = 1  # Default user for single-user mode
    trading_mode: TradingMode = TradingMode.MANUAL
    risk_profile: RiskProfile = RiskProfile.MODERATE

    # Confidence thresholds
    min_confidence_auto: float = 0.75   # Min confidence for auto-execution
    min_confidence_manual: float = 0.50  # Min confidence to show in queue
    min_consensus: float = 0.60          # Min consensus from signal sources

    # Position limits
    max_position_size_usd: float = 1000.0   # Max per-trade
    max_portfolio_pct: float = 0.10          # Max % of portfolio per position
    max_open_positions: int = 5              # Max concurrent positions

    # Daily limits
    max_trades_per_day: int = 20
    max_daily_loss_usd: float = 500.0        # Stop trading if hit

    # Symbol restrictions
    allowed_symbols: List[str] = None        # None = all allowed
    blocked_symbols: List[str] = None        # Explicit blocklist

    # Timing
    signal_expiry_minutes: int = 15          # How long before signals expire
    cooldown_minutes: int = 5                # Min time between trades on same symbol

    # Emergency controls
    kill_switch_active: bool = False         # Master kill switch
    kill_switch_reason: str = ""
    kill_switch_timestamp: str = None

    # Paper trading mode
    paper_trading: bool = True               # Default to paper trading for safety

    # Timestamps
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.allowed_symbols is None:
            self.allowed_symbols = []
        if self.blocked_symbols is None:
            self.blocked_symbols = []
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow().isoformat()


@dataclass
class PendingSignal:
    """A trading signal awaiting action"""
    id: int = None
    signal_source: str = ""          # 'tradingview', 'ai_prediction', 'rl_agent', etc.
    symbol: str = ""
    action: str = ""                 # 'BUY', 'SELL', 'HOLD'
    confidence: float = 0.0
    consensus: float = 0.0
    recommended_size_usd: float = 0.0
    recommended_size_pct: float = 0.0

    # Signal details
    signal_data: Dict = None         # Full signal payload
    aggregation_data: Dict = None    # Signal aggregation details

    # Market context
    current_price: float = 0.0
    market_context: Dict = None

    # Status tracking
    status: SignalStatus = SignalStatus.PENDING
    status_reason: str = ""

    # Execution details
    order_id: int = None
    executed_price: float = None
    executed_qty: float = None
    execution_error: str = None

    # Timestamps
    created_at: str = None
    expires_at: str = None
    actioned_at: str = None

    def __post_init__(self):
        if self.signal_data is None:
            self.signal_data = {}
        if self.aggregation_data is None:
            self.aggregation_data = {}
        if self.market_context is None:
            self.market_context = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class DailyStats:
    """Daily trading statistics for risk management"""
    date: str
    trades_count: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    volume_usd: float = 0.0


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_trading_schema():
    """Initialize autonomous trading database schema."""
    conn = get_db()
    cursor = conn.cursor()

    # Trading settings per user
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trading_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            trading_mode TEXT DEFAULT 'manual',
            risk_profile TEXT DEFAULT 'moderate',

            min_confidence_auto REAL DEFAULT 0.75,
            min_confidence_manual REAL DEFAULT 0.50,
            min_consensus REAL DEFAULT 0.60,

            max_position_size_usd REAL DEFAULT 1000.0,
            max_portfolio_pct REAL DEFAULT 0.10,
            max_open_positions INTEGER DEFAULT 5,

            max_trades_per_day INTEGER DEFAULT 20,
            max_daily_loss_usd REAL DEFAULT 500.0,

            allowed_symbols TEXT DEFAULT '[]',
            blocked_symbols TEXT DEFAULT '[]',

            signal_expiry_minutes INTEGER DEFAULT 15,
            cooldown_minutes INTEGER DEFAULT 5,

            kill_switch_active BOOLEAN DEFAULT 0,
            kill_switch_reason TEXT DEFAULT '',
            kill_switch_timestamp TEXT,

            paper_trading BOOLEAN DEFAULT 1,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(user_id)
        )
    """)

    # Pending signals queue
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            signal_source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            confidence REAL NOT NULL,
            consensus REAL DEFAULT 0,

            recommended_size_usd REAL,
            recommended_size_pct REAL,

            signal_data TEXT DEFAULT '{}',
            aggregation_data TEXT DEFAULT '{}',

            current_price REAL,
            market_context TEXT DEFAULT '{}',

            status TEXT DEFAULT 'pending',
            status_reason TEXT DEFAULT '',

            order_id INTEGER,
            executed_price REAL,
            executed_qty REAL,
            execution_error TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            actioned_at TIMESTAMP,

            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_signals_status ON pending_signals(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_signals_symbol ON pending_signals(symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_signals_user ON pending_signals(user_id)")

    # Daily trading stats for risk management
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_trading_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            date TEXT NOT NULL,
            trades_count INTEGER DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            volume_usd REAL DEFAULT 0,

            UNIQUE(user_id, date)
        )
    """)

    # Trading activity log for audit
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trading_activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            activity_type TEXT NOT NULL,
            details TEXT DEFAULT '{}',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Symbol cooldown tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbol_cooldowns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            symbol TEXT NOT NULL,
            last_trade_at TIMESTAMP NOT NULL,

            UNIQUE(user_id, symbol)
        )
    """)

    conn.commit()
    conn.close()
    print("[OK] Autonomous trading schema initialized")


class TradingSettingsManager:
    """Manage user trading settings"""

    @staticmethod
    def get_settings(user_id: int = 1) -> TradingSettings:
        """Get trading settings for a user, creating defaults if needed."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM trading_settings WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if not row:
            # Create default settings
            cursor.execute("""
                INSERT INTO trading_settings (user_id) VALUES (?)
            """, (user_id,))
            conn.commit()
            cursor.execute("SELECT * FROM trading_settings WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

        conn.close()

        return TradingSettings(
            user_id=row['user_id'],
            trading_mode=TradingMode(row['trading_mode']),
            risk_profile=RiskProfile(row['risk_profile']),
            min_confidence_auto=row['min_confidence_auto'],
            min_confidence_manual=row['min_confidence_manual'],
            min_consensus=row['min_consensus'],
            max_position_size_usd=row['max_position_size_usd'],
            max_portfolio_pct=row['max_portfolio_pct'],
            max_open_positions=row['max_open_positions'],
            max_trades_per_day=row['max_trades_per_day'],
            max_daily_loss_usd=row['max_daily_loss_usd'],
            allowed_symbols=json.loads(row['allowed_symbols'] or '[]'),
            blocked_symbols=json.loads(row['blocked_symbols'] or '[]'),
            signal_expiry_minutes=row['signal_expiry_minutes'],
            cooldown_minutes=row['cooldown_minutes'],
            kill_switch_active=bool(row['kill_switch_active']),
            kill_switch_reason=row['kill_switch_reason'] or '',
            kill_switch_timestamp=row['kill_switch_timestamp'],
            paper_trading=bool(row['paper_trading']),
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    @staticmethod
    def update_settings(user_id: int, updates: Dict[str, Any]) -> TradingSettings:
        """Update trading settings for a user."""
        conn = get_db()
        cursor = conn.cursor()

        # Build update query
        allowed_fields = [
            'trading_mode', 'risk_profile', 'min_confidence_auto', 'min_confidence_manual',
            'min_consensus', 'max_position_size_usd', 'max_portfolio_pct', 'max_open_positions',
            'max_trades_per_day', 'max_daily_loss_usd', 'allowed_symbols', 'blocked_symbols',
            'signal_expiry_minutes', 'cooldown_minutes', 'paper_trading'
        ]

        set_clauses = []
        values = []

        for field, value in updates.items():
            if field in allowed_fields:
                if field in ['allowed_symbols', 'blocked_symbols']:
                    value = json.dumps(value) if isinstance(value, list) else value
                set_clauses.append(f"{field} = ?")
                values.append(value)

        if set_clauses:
            set_clauses.append("updated_at = ?")
            values.append(datetime.utcnow().isoformat())
            values.append(user_id)

            query = f"UPDATE trading_settings SET {', '.join(set_clauses)} WHERE user_id = ?"
            cursor.execute(query, values)
            conn.commit()

            # Log the change
            TradingActivityLogger.log(user_id, "settings_updated", updates)

        conn.close()
        return TradingSettingsManager.get_settings(user_id)

    @staticmethod
    def activate_kill_switch(user_id: int, reason: str = "Manual activation") -> TradingSettings:
        """Activate emergency kill switch."""
        conn = get_db()
        cursor = conn.cursor()

        timestamp = datetime.utcnow().isoformat()
        cursor.execute("""
            UPDATE trading_settings
            SET kill_switch_active = 1, kill_switch_reason = ?, kill_switch_timestamp = ?, updated_at = ?
            WHERE user_id = ?
        """, (reason, timestamp, timestamp, user_id))
        conn.commit()
        conn.close()

        TradingActivityLogger.log(user_id, "kill_switch_activated", {"reason": reason})
        return TradingSettingsManager.get_settings(user_id)

    @staticmethod
    def deactivate_kill_switch(user_id: int) -> TradingSettings:
        """Deactivate kill switch."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE trading_settings
            SET kill_switch_active = 0, kill_switch_reason = '', kill_switch_timestamp = NULL, updated_at = ?
            WHERE user_id = ?
        """, (datetime.utcnow().isoformat(), user_id))
        conn.commit()
        conn.close()

        TradingActivityLogger.log(user_id, "kill_switch_deactivated", {})
        return TradingSettingsManager.get_settings(user_id)


class TradingActivityLogger:
    """Log trading activities for audit"""

    @staticmethod
    def log(user_id: int, activity_type: str, details: Dict = None):
        """Log a trading activity."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trading_activity_log (user_id, activity_type, details)
            VALUES (?, ?, ?)
        """, (user_id, activity_type, json.dumps(details or {})))
        conn.commit()
        conn.close()

    @staticmethod
    def get_recent(user_id: int, limit: int = 50) -> List[Dict]:
        """Get recent activity logs."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trading_activity_log
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]


class DailyStatsManager:
    """Manage daily trading statistics"""

    @staticmethod
    def get_today(user_id: int = 1) -> DailyStats:
        """Get today's trading stats."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM daily_trading_stats WHERE user_id = ? AND date = ?
        """, (user_id, today))
        row = cursor.fetchone()

        if not row:
            cursor.execute("""
                INSERT INTO daily_trading_stats (user_id, date) VALUES (?, ?)
            """, (user_id, today))
            conn.commit()
            return DailyStats(date=today)

        conn.close()
        return DailyStats(
            date=row['date'],
            trades_count=row['trades_count'],
            realized_pnl=row['realized_pnl'],
            unrealized_pnl=row['unrealized_pnl'],
            wins=row['wins'],
            losses=row['losses'],
            volume_usd=row['volume_usd']
        )

    @staticmethod
    def increment_trade(user_id: int, volume_usd: float, pnl: float = 0):
        """Record a trade."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        conn = get_db()
        cursor = conn.cursor()

        win_inc = 1 if pnl > 0 else 0
        loss_inc = 1 if pnl < 0 else 0

        cursor.execute("""
            INSERT INTO daily_trading_stats (user_id, date, trades_count, realized_pnl, volume_usd, wins, losses)
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET
                trades_count = trades_count + 1,
                realized_pnl = realized_pnl + ?,
                volume_usd = volume_usd + ?,
                wins = wins + ?,
                losses = losses + ?
        """, (user_id, today, pnl, volume_usd, win_inc, loss_inc, pnl, volume_usd, win_inc, loss_inc))
        conn.commit()
        conn.close()


class PendingSignalQueue:
    """Manage the pending signals queue"""

    @staticmethod
    def add_signal(signal: PendingSignal) -> int:
        """Add a signal to the queue."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pending_signals (
                user_id, signal_source, symbol, action, confidence, consensus,
                recommended_size_usd, recommended_size_pct, signal_data, aggregation_data,
                current_price, market_context, status, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.user_id if hasattr(signal, 'user_id') else 1,
            signal.signal_source,
            signal.symbol,
            signal.action,
            signal.confidence,
            signal.consensus,
            signal.recommended_size_usd,
            signal.recommended_size_pct,
            json.dumps(signal.signal_data),
            json.dumps(signal.aggregation_data),
            signal.current_price,
            json.dumps(signal.market_context),
            signal.status.value if isinstance(signal.status, SignalStatus) else signal.status,
            signal.expires_at
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return signal_id

    @staticmethod
    def get_pending(user_id: int = 1, include_expired: bool = False) -> List[PendingSignal]:
        """Get all pending signals for a user."""
        conn = get_db()
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        if include_expired:
            cursor.execute("""
                SELECT * FROM pending_signals
                WHERE user_id = ? AND status = 'pending'
                ORDER BY created_at DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT * FROM pending_signals
                WHERE user_id = ? AND status = 'pending'
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY confidence DESC, created_at DESC
            """, (user_id, now))

        rows = cursor.fetchall()
        conn.close()

        signals = []
        for row in rows:
            signals.append(PendingSignal(
                id=row['id'],
                signal_source=row['signal_source'],
                symbol=row['symbol'],
                action=row['action'],
                confidence=row['confidence'],
                consensus=row['consensus'],
                recommended_size_usd=row['recommended_size_usd'],
                recommended_size_pct=row['recommended_size_pct'],
                signal_data=json.loads(row['signal_data'] or '{}'),
                aggregation_data=json.loads(row['aggregation_data'] or '{}'),
                current_price=row['current_price'],
                market_context=json.loads(row['market_context'] or '{}'),
                status=SignalStatus(row['status']),
                status_reason=row['status_reason'] or '',
                order_id=row['order_id'],
                executed_price=row['executed_price'],
                executed_qty=row['executed_qty'],
                execution_error=row['execution_error'],
                created_at=row['created_at'],
                expires_at=row['expires_at'],
                actioned_at=row['actioned_at']
            ))

        return signals

    @staticmethod
    def get_signal(signal_id: int) -> Optional[PendingSignal]:
        """Get a specific signal by ID."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pending_signals WHERE id = ?", (signal_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return PendingSignal(
            id=row['id'],
            signal_source=row['signal_source'],
            symbol=row['symbol'],
            action=row['action'],
            confidence=row['confidence'],
            consensus=row['consensus'],
            recommended_size_usd=row['recommended_size_usd'],
            recommended_size_pct=row['recommended_size_pct'],
            signal_data=json.loads(row['signal_data'] or '{}'),
            aggregation_data=json.loads(row['aggregation_data'] or '{}'),
            current_price=row['current_price'],
            market_context=json.loads(row['market_context'] or '{}'),
            status=SignalStatus(row['status']),
            status_reason=row['status_reason'] or '',
            order_id=row['order_id'],
            executed_price=row['executed_price'],
            executed_qty=row['executed_qty'],
            execution_error=row['execution_error'],
            created_at=row['created_at'],
            expires_at=row['expires_at'],
            actioned_at=row['actioned_at']
        )

    @staticmethod
    def update_status(signal_id: int, status: SignalStatus, reason: str = "",
                      order_id: int = None, executed_price: float = None,
                      executed_qty: float = None, execution_error: str = None):
        """Update signal status."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE pending_signals
            SET status = ?, status_reason = ?, order_id = ?, executed_price = ?,
                executed_qty = ?, execution_error = ?, actioned_at = ?
            WHERE id = ?
        """, (
            status.value if isinstance(status, SignalStatus) else status,
            reason,
            order_id,
            executed_price,
            executed_qty,
            execution_error,
            datetime.utcnow().isoformat(),
            signal_id
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def expire_old_signals(user_id: int = 1):
        """Mark expired signals."""
        conn = get_db()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute("""
            UPDATE pending_signals
            SET status = 'expired', actioned_at = ?
            WHERE user_id = ? AND status = 'pending' AND expires_at < ?
        """, (now, user_id, now))

        expired_count = cursor.rowcount
        conn.commit()
        conn.close()

        return expired_count

    @staticmethod
    def get_history(user_id: int = 1, limit: int = 100) -> List[PendingSignal]:
        """Get signal history (non-pending)."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM pending_signals
            WHERE user_id = ? AND status != 'pending'
            ORDER BY actioned_at DESC
            LIMIT ?
        """, (user_id, limit))

        rows = cursor.fetchall()
        conn.close()

        signals = []
        for row in rows:
            signals.append(PendingSignal(
                id=row['id'],
                signal_source=row['signal_source'],
                symbol=row['symbol'],
                action=row['action'],
                confidence=row['confidence'],
                consensus=row['consensus'],
                recommended_size_usd=row['recommended_size_usd'],
                recommended_size_pct=row['recommended_size_pct'],
                signal_data=json.loads(row['signal_data'] or '{}'),
                aggregation_data=json.loads(row['aggregation_data'] or '{}'),
                current_price=row['current_price'],
                market_context=json.loads(row['market_context'] or '{}'),
                status=SignalStatus(row['status']),
                status_reason=row['status_reason'] or '',
                order_id=row['order_id'],
                executed_price=row['executed_price'],
                executed_qty=row['executed_qty'],
                execution_error=row['execution_error'],
                created_at=row['created_at'],
                expires_at=row['expires_at'],
                actioned_at=row['actioned_at']
            ))

        return signals


class CooldownManager:
    """Manage trading cooldowns per symbol"""

    @staticmethod
    def is_on_cooldown(user_id: int, symbol: str, cooldown_minutes: int) -> bool:
        """Check if symbol is on cooldown."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT last_trade_at FROM symbol_cooldowns
            WHERE user_id = ? AND symbol = ?
        """, (user_id, symbol))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False

        last_trade = datetime.fromisoformat(row['last_trade_at'])
        cooldown_until = last_trade + timedelta(minutes=cooldown_minutes)

        return datetime.utcnow() < cooldown_until

    @staticmethod
    def set_cooldown(user_id: int, symbol: str):
        """Set cooldown for a symbol."""
        conn = get_db()
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO symbol_cooldowns (user_id, symbol, last_trade_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, symbol) DO UPDATE SET last_trade_at = ?
        """, (user_id, symbol, now, now))
        conn.commit()
        conn.close()


class RiskManager:
    """Risk management checks before execution"""

    @staticmethod
    def check_can_trade(user_id: int, symbol: str, size_usd: float, settings: TradingSettings = None) -> Dict[str, Any]:
        """
        Comprehensive risk check before allowing a trade.
        Returns dict with 'allowed' bool and 'reasons' list.
        """
        if settings is None:
            settings = TradingSettingsManager.get_settings(user_id)

        reasons = []

        # Kill switch check
        if settings.kill_switch_active:
            return {
                "allowed": False,
                "reasons": [f"Kill switch active: {settings.kill_switch_reason}"]
            }

        # Trading mode check
        if settings.trading_mode == TradingMode.OFF:
            return {
                "allowed": False,
                "reasons": ["Trading is disabled"]
            }

        # Symbol checks
        if settings.blocked_symbols and symbol in settings.blocked_symbols:
            reasons.append(f"Symbol {symbol} is blocked")

        if settings.allowed_symbols and symbol not in settings.allowed_symbols:
            if settings.allowed_symbols:  # Only check if allowlist is not empty
                reasons.append(f"Symbol {symbol} not in allowed list")

        # Position size check
        if size_usd > settings.max_position_size_usd:
            reasons.append(f"Position size ${size_usd:.2f} exceeds max ${settings.max_position_size_usd:.2f}")

        # Daily stats checks
        daily_stats = DailyStatsManager.get_today(user_id)

        if daily_stats.trades_count >= settings.max_trades_per_day:
            reasons.append(f"Daily trade limit reached ({daily_stats.trades_count}/{settings.max_trades_per_day})")

        if daily_stats.realized_pnl <= -settings.max_daily_loss_usd:
            reasons.append(f"Daily loss limit reached (${daily_stats.realized_pnl:.2f})")

        # Cooldown check
        if CooldownManager.is_on_cooldown(user_id, symbol, settings.cooldown_minutes):
            reasons.append(f"Symbol {symbol} is on cooldown")

        # TODO: Add position count check (requires broker integration)
        # TODO: Add portfolio percentage check (requires account balance)

        return {
            "allowed": len(reasons) == 0,
            "reasons": reasons
        }

    @staticmethod
    def calculate_position_size(confidence: float, settings: TradingSettings,
                                account_balance: float = None) -> Dict[str, float]:
        """
        Calculate recommended position size based on confidence and risk profile.
        """
        # Base size multipliers by risk profile
        profile_multipliers = {
            RiskProfile.CONSERVATIVE: 0.5,
            RiskProfile.MODERATE: 1.0,
            RiskProfile.AGGRESSIVE: 1.5
        }

        multiplier = profile_multipliers.get(settings.risk_profile, 1.0)

        # Confidence-based scaling (higher confidence = larger position)
        confidence_scale = 0.5 + (confidence * 0.5)  # 0.5 to 1.0

        # Calculate base size
        base_size = settings.max_position_size_usd * multiplier * confidence_scale

        # Cap at max
        size_usd = min(base_size, settings.max_position_size_usd)

        # Calculate percentage if account balance provided
        size_pct = 0.0
        if account_balance and account_balance > 0:
            size_pct = min(size_usd / account_balance, settings.max_portfolio_pct)
            size_usd = min(size_usd, account_balance * settings.max_portfolio_pct)

        return {
            "size_usd": round(size_usd, 2),
            "size_pct": round(size_pct, 4),
            "confidence_factor": round(confidence_scale, 2),
            "risk_multiplier": multiplier
        }


class AutonomousTradingEngine:
    """
    Main autonomous trading engine that processes signals and executes trades.
    """

    def __init__(self, user_id: int = 1):
        self.user_id = user_id
        self._running = False
        self._loop_task = None

    async def process_aggregated_signal(self, aggregation_result: Dict) -> Dict[str, Any]:
        """
        Process an aggregated signal from the signal aggregator.
        Decides whether to auto-execute, queue for approval, or reject.
        """
        settings = TradingSettingsManager.get_settings(self.user_id)

        # Extract signal details
        symbol = aggregation_result.get('symbol', '')
        action = aggregation_result.get('action', 'HOLD')
        confidence = aggregation_result.get('confidence', 0)
        consensus = aggregation_result.get('consensus', 0)
        should_execute = aggregation_result.get('should_execute', False)
        current_price = aggregation_result.get('market_context', {}).get('current_price', 0)

        # Skip HOLD signals
        if action == 'HOLD':
            return {
                "action": "skipped",
                "reason": "HOLD signal - no action needed",
                "signal_id": None
            }

        # Calculate position size
        sizing = RiskManager.calculate_position_size(confidence, settings)

        # Create pending signal
        expiry_time = datetime.utcnow() + timedelta(minutes=settings.signal_expiry_minutes)

        signal = PendingSignal(
            signal_source=aggregation_result.get('source', 'aggregator'),
            symbol=symbol,
            action=action,
            confidence=confidence,
            consensus=consensus,
            recommended_size_usd=sizing['size_usd'],
            recommended_size_pct=sizing['size_pct'],
            signal_data=aggregation_result.get('signal_data', {}),
            aggregation_data=aggregation_result,
            current_price=current_price,
            market_context=aggregation_result.get('market_context', {}),
            expires_at=expiry_time.isoformat()
        )

        # Risk checks
        risk_check = RiskManager.check_can_trade(self.user_id, symbol, sizing['size_usd'], settings)

        if not risk_check['allowed']:
            signal.status = SignalStatus.REJECTED
            signal.status_reason = "; ".join(risk_check['reasons'])
            signal_id = PendingSignalQueue.add_signal(signal)

            TradingActivityLogger.log(self.user_id, "signal_rejected", {
                "signal_id": signal_id,
                "symbol": symbol,
                "reasons": risk_check['reasons']
            })

            return {
                "action": "rejected",
                "reason": signal.status_reason,
                "signal_id": signal_id
            }

        # Determine action based on trading mode and confidence
        if settings.trading_mode == TradingMode.FULL_AUTO:
            if confidence >= settings.min_confidence_auto and consensus >= settings.min_consensus:
                signal.status = SignalStatus.AUTO_APPROVED
                signal_id = PendingSignalQueue.add_signal(signal)

                # Execute immediately
                execution_result = await self.execute_signal(signal_id)
                return {
                    "action": "auto_executed",
                    "signal_id": signal_id,
                    "execution": execution_result
                }

        elif settings.trading_mode == TradingMode.SEMI_AUTO:
            if confidence >= settings.min_confidence_auto and consensus >= settings.min_consensus:
                signal.status = SignalStatus.AUTO_APPROVED
                signal_id = PendingSignalQueue.add_signal(signal)

                execution_result = await self.execute_signal(signal_id)
                return {
                    "action": "auto_executed",
                    "signal_id": signal_id,
                    "execution": execution_result
                }
            elif confidence >= settings.min_confidence_manual:
                # Queue for manual approval
                signal.status = SignalStatus.PENDING
                signal_id = PendingSignalQueue.add_signal(signal)

                return {
                    "action": "queued",
                    "reason": f"Confidence {confidence:.1%} below auto threshold {settings.min_confidence_auto:.1%}",
                    "signal_id": signal_id
                }

        elif settings.trading_mode == TradingMode.MANUAL:
            if confidence >= settings.min_confidence_manual:
                signal.status = SignalStatus.PENDING
                signal_id = PendingSignalQueue.add_signal(signal)

                return {
                    "action": "queued",
                    "reason": "Manual approval required",
                    "signal_id": signal_id
                }

        # Below minimum confidence threshold
        signal.status = SignalStatus.REJECTED
        signal.status_reason = f"Confidence {confidence:.1%} below minimum {settings.min_confidence_manual:.1%}"
        signal_id = PendingSignalQueue.add_signal(signal)

        return {
            "action": "rejected",
            "reason": signal.status_reason,
            "signal_id": signal_id
        }

    async def execute_signal(self, signal_id: int) -> Dict[str, Any]:
        """Execute a trading signal."""
        signal = PendingSignalQueue.get_signal(signal_id)
        if not signal:
            return {"success": False, "error": "Signal not found"}

        if signal.status not in [SignalStatus.PENDING, SignalStatus.APPROVED, SignalStatus.AUTO_APPROVED]:
            return {"success": False, "error": f"Signal status {signal.status} cannot be executed"}

        settings = TradingSettingsManager.get_settings(self.user_id)

        # Final risk check
        risk_check = RiskManager.check_can_trade(
            self.user_id, signal.symbol, signal.recommended_size_usd, settings
        )

        if not risk_check['allowed']:
            PendingSignalQueue.update_status(
                signal_id, SignalStatus.REJECTED,
                reason="; ".join(risk_check['reasons'])
            )
            return {"success": False, "error": risk_check['reasons']}

        # Mark as executing
        PendingSignalQueue.update_status(signal_id, SignalStatus.EXECUTING)

        try:
            # Import order service
            from backend.order_service import OrderExecutionService

            order_service = OrderExecutionService()

            # Build order payload
            order_payload = {
                "symbol": signal.symbol,
                "side": signal.action.lower(),
                "notional": signal.recommended_size_usd,
                "order_type": "market",
                "time_in_force": "day"
            }

            # Execute via broker
            if settings.paper_trading:
                # Use paper trading mode
                result = await order_service.execute_webhook_order(
                    order_payload,
                    broker_type="alpaca"
                )
            else:
                result = await order_service.execute_webhook_order(
                    order_payload,
                    broker_type="alpaca"
                )

            if result.get('success'):
                # Update signal with execution details
                PendingSignalQueue.update_status(
                    signal_id,
                    SignalStatus.EXECUTED,
                    reason="Successfully executed",
                    order_id=result.get('order_id'),
                    executed_price=result.get('filled_avg_price'),
                    executed_qty=result.get('filled_qty')
                )

                # Set cooldown
                CooldownManager.set_cooldown(self.user_id, signal.symbol)

                # Update daily stats
                DailyStatsManager.increment_trade(self.user_id, signal.recommended_size_usd)

                # Log activity
                TradingActivityLogger.log(self.user_id, "trade_executed", {
                    "signal_id": signal_id,
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "size_usd": signal.recommended_size_usd,
                    "order_id": result.get('order_id')
                })

                return {"success": True, "order": result}
            else:
                PendingSignalQueue.update_status(
                    signal_id,
                    SignalStatus.FAILED,
                    execution_error=result.get('error', 'Unknown error')
                )
                return {"success": False, "error": result.get('error')}

        except Exception as e:
            error_msg = str(e)
            PendingSignalQueue.update_status(
                signal_id,
                SignalStatus.FAILED,
                execution_error=error_msg
            )
            return {"success": False, "error": error_msg}

    async def approve_signal(self, signal_id: int) -> Dict[str, Any]:
        """Manually approve and execute a pending signal."""
        signal = PendingSignalQueue.get_signal(signal_id)
        if not signal:
            return {"success": False, "error": "Signal not found"}

        if signal.status != SignalStatus.PENDING:
            return {"success": False, "error": f"Signal is {signal.status}, not pending"}

        PendingSignalQueue.update_status(signal_id, SignalStatus.APPROVED, reason="Manual approval")
        TradingActivityLogger.log(self.user_id, "signal_approved", {"signal_id": signal_id})

        return await self.execute_signal(signal_id)

    async def reject_signal(self, signal_id: int, reason: str = "Manual rejection") -> Dict[str, Any]:
        """Manually reject a pending signal."""
        signal = PendingSignalQueue.get_signal(signal_id)
        if not signal:
            return {"success": False, "error": "Signal not found"}

        PendingSignalQueue.update_status(signal_id, SignalStatus.REJECTED, reason=reason)
        TradingActivityLogger.log(self.user_id, "signal_rejected", {
            "signal_id": signal_id,
            "reason": reason
        })

        return {"success": True, "message": "Signal rejected"}

    async def run_autonomous_loop(self, interval_seconds: int = 30):
        """
        Run the autonomous trading loop.
        This periodically:
        1. Expires old signals
        2. Checks for new signals from AI models
        3. Processes signals based on trading mode
        """
        self._running = True

        TradingActivityLogger.log(self.user_id, "autonomous_loop_started", {
            "interval": interval_seconds
        })

        while self._running:
            try:
                settings = TradingSettingsManager.get_settings(self.user_id)

                # Skip if kill switch active or trading off
                if settings.kill_switch_active or settings.trading_mode == TradingMode.OFF:
                    await asyncio.sleep(interval_seconds)
                    continue

                # Expire old signals
                expired = PendingSignalQueue.expire_old_signals(self.user_id)
                if expired > 0:
                    TradingActivityLogger.log(self.user_id, "signals_expired", {"count": expired})

                # TODO: Poll AI models for new predictions
                # This would integrate with the AI prediction endpoints

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                TradingActivityLogger.log(self.user_id, "autonomous_loop_error", {
                    "error": str(e)
                })
                await asyncio.sleep(interval_seconds)

        TradingActivityLogger.log(self.user_id, "autonomous_loop_stopped", {})

    def stop_loop(self):
        """Stop the autonomous trading loop."""
        self._running = False


# Initialize schema when module is imported
try:
    init_trading_schema()
except Exception as e:
    print(f"[WARN] Could not initialize trading schema: {e}")
