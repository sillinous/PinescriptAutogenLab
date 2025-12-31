# backend/database.py

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import os

# Import encryption utilities
try:
    from backend.security.encryption import encrypt_credential, decrypt_credential
    ENCRYPTION_AVAILABLE = True
except (ImportError, ValueError) as e:
    # Encryption not configured - warn but don't fail
    print(f"[WARN] Encryption not available: {e}")
    print("[WARN] Credentials will be stored in plaintext. Set ENCRYPTION_KEY in .env for production.")
    ENCRYPTION_AVAILABLE = False

    def encrypt_credential(text: str) -> str:
        return text

    def decrypt_credential(text: str) -> str:
        return text

# Database location - configurable via env
DATA_DIR = Path(os.getenv("PINELAB_DATA", "./data"))
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "pinelab.db"


def get_db():
    """Get database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_db()
    cursor = conn.cursor()

    # Broker credentials table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS broker_credentials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_type TEXT NOT NULL,  -- 'alpaca', 'ccxt'
            exchange TEXT,  -- NULL for alpaca, 'binance'/'coinbase' for ccxt
            api_key TEXT NOT NULL,
            api_secret TEXT NOT NULL,
            api_passphrase TEXT,  -- for some exchanges
            paper_trading BOOLEAN DEFAULT 1,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_order_id TEXT UNIQUE,
            broker_order_id TEXT,
            broker_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,  -- 'buy', 'sell'
            order_type TEXT DEFAULT 'market',  -- 'market', 'limit'
            qty REAL,
            notional REAL,  -- for notional orders
            limit_price REAL,
            filled_qty REAL DEFAULT 0,
            filled_avg_price REAL,
            status TEXT DEFAULT 'pending',  -- 'pending', 'submitted', 'filled', 'partial', 'cancelled', 'rejected'
            time_in_force TEXT DEFAULT 'day',
            webhook_payload TEXT,  -- original TradingView JSON
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submitted_at TIMESTAMP,
            filled_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Fills/executions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            broker_fill_id TEXT,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            commission REAL DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    """)

    # NEW: Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            full_name TEXT,
            is_active BOOLEAN DEFAULT 1,
            is_admin BOOLEAN DEFAULT 0,
            is_verified BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)

    # Add user_id to orders table
    try:
        cursor.execute("ALTER TABLE orders ADD COLUMN user_id INTEGER REFERENCES users(id)")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise

    # Positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_entry_price REAL NOT NULL,
            current_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            cost_basis REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(broker_type, symbol)
        )
    """)

    # Strategy parameters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            param_name TEXT NOT NULL,
            param_value TEXT NOT NULL,
            is_best BOOLEAN DEFAULT 0,  -- promoted from optimization
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(strategy_name, param_name)
        )
    """)

    # Performance tracking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            portfolio_value REAL NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            gross_profit REAL DEFAULT 0,
            gross_loss REAL DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Webhook log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS webhook_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,  -- 'tradingview', 'manual', etc.
            payload TEXT NOT NULL,
            signature TEXT,
            signature_valid BOOLEAN,
            ip_address TEXT,
            order_id INTEGER,  -- NULL if rejected before order creation
            status TEXT DEFAULT 'received',  -- 'received', 'processed', 'rejected'
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_webhook_timestamp ON webhook_log(timestamp)")

    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at {DB_PATH}")


# --- Database helper functions ---

def insert_order(data: Dict[str, Any]) -> int:
    """Insert new order and return order ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO orders (
            client_order_id, broker_type, symbol, side, order_type,
            qty, notional, limit_price, webhook_payload, time_in_force
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get('client_order_id'),
        data.get('broker_type'),
        data.get('symbol'),
        data.get('side'),
        data.get('order_type', 'market'),
        data.get('qty'),
        data.get('notional'),
        data.get('limit_price'),
        data.get('webhook_payload'),
        data.get('time_in_force', 'day')
    ))

    order_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return order_id


def update_order(order_id: int, updates: Dict[str, Any]):
    """Update order with new data."""
    conn = get_db()
    cursor = conn.cursor()

    # Build dynamic UPDATE query
    set_clauses = []
    values = []
    for key, value in updates.items():
        set_clauses.append(f"{key} = ?")
        values.append(value)

    values.append(order_id)
    query = f"UPDATE orders SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"

    cursor.execute(query, values)
    conn.commit()
    conn.close()


def get_order_by_id(order_id: int) -> Optional[Dict]:
    """Retrieve order by ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_orders(limit: int = 100) -> List[Dict]:
    """Get recent orders."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM orders
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def insert_fill(order_id: int, qty: float, price: float, commission: float = 0, broker_fill_id: str = None):
    """Record a fill/execution."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO fills (order_id, broker_fill_id, qty, price, commission)
        VALUES (?, ?, ?, ?, ?)
    """, (order_id, broker_fill_id, qty, price, commission))

    conn.commit()
    conn.close()


def upsert_position(broker_type: str, symbol: str, qty: float, avg_entry_price: float,
                    current_price: float = None, unrealized_pnl: float = None):
    """Update or insert position."""
    conn = get_db()
    cursor = conn.cursor()

    market_value = (qty * current_price) if current_price else None
    cost_basis = qty * avg_entry_price
    unrealized_pnl_pct = ((current_price - avg_entry_price) / avg_entry_price * 100) if current_price and avg_entry_price else None

    cursor.execute("""
        INSERT INTO positions (broker_type, symbol, qty, avg_entry_price, current_price, market_value, unrealized_pnl, unrealized_pnl_pct, cost_basis)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(broker_type, symbol) DO UPDATE SET
            qty = excluded.qty,
            avg_entry_price = excluded.avg_entry_price,
            current_price = excluded.current_price,
            market_value = excluded.market_value,
            unrealized_pnl = excluded.unrealized_pnl,
            unrealized_pnl_pct = excluded.unrealized_pnl_pct,
            cost_basis = excluded.cost_basis,
            updated_at = CURRENT_TIMESTAMP
    """, (broker_type, symbol, qty, avg_entry_price, current_price, market_value, unrealized_pnl, unrealized_pnl_pct, cost_basis))

    conn.commit()
    conn.close()


def get_all_positions() -> List[Dict]:
    """Get all current positions."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM positions WHERE qty != 0")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_webhook(payload: str, signature: str = None, signature_valid: bool = None,
                ip_address: str = None, order_id: int = None, status: str = 'received',
                error_message: str = None, source: str = 'tradingview') -> int:
    """Log incoming webhook."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO webhook_log (source, payload, signature, signature_valid, ip_address, order_id, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (source, payload, signature, signature_valid, ip_address, order_id, status, error_message))

    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id


def save_broker_credentials(broker_type: str, api_key: str, api_secret: str,
                            exchange: str = None, paper_trading: bool = True,
                            api_passphrase: str = None) -> int:
    """
    Save broker credentials with encryption.

    Credentials are encrypted using AES-256 before storage.
    Requires ENCRYPTION_KEY environment variable to be set.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Encrypt sensitive fields
    encrypted_api_key = encrypt_credential(api_key)
    encrypted_api_secret = encrypt_credential(api_secret)
    encrypted_passphrase = encrypt_credential(api_passphrase) if api_passphrase else None

    # Deactivate old credentials for same broker/exchange
    cursor.execute("""
        UPDATE broker_credentials
        SET is_active = 0
        WHERE broker_type = ? AND (exchange = ? OR (exchange IS NULL AND ? IS NULL))
    """, (broker_type, exchange, exchange))

    # Insert new credentials (encrypted)
    cursor.execute("""
        INSERT INTO broker_credentials (broker_type, exchange, api_key, api_secret, api_passphrase, paper_trading)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (broker_type, exchange, encrypted_api_key, encrypted_api_secret, encrypted_passphrase, paper_trading))

    cred_id = cursor.lastrowid
    conn.commit()
    conn.close()

    if ENCRYPTION_AVAILABLE:
        print(f"[OK] Saved encrypted credentials for {broker_type}" + (f"/{exchange}" if exchange else ""))
    else:
        print(f"[WARN] Saved UNENCRYPTED credentials for {broker_type}. Set ENCRYPTION_KEY for security!")

    return cred_id


def get_active_broker_credentials(broker_type: str, exchange: str = None) -> Optional[Dict]:
    """
    Get active broker credentials with decryption.

    Returns decrypted credentials ready for use.
    """
    conn = get_db()
    cursor = conn.cursor()

    if exchange:
        cursor.execute("""
            SELECT * FROM broker_credentials
            WHERE broker_type = ? AND exchange = ? AND is_active = 1
            ORDER BY created_at DESC LIMIT 1
        """, (broker_type, exchange))
    else:
        cursor.execute("""
            SELECT * FROM broker_credentials
            WHERE broker_type = ? AND is_active = 1
            ORDER BY created_at DESC LIMIT 1
        """, (broker_type,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    # Convert to dict and decrypt sensitive fields
    creds = dict(row)

    try:
        creds['api_key'] = decrypt_credential(creds['api_key'])
        creds['api_secret'] = decrypt_credential(creds['api_secret'])
        if creds.get('api_passphrase'):
            creds['api_passphrase'] = decrypt_credential(creds['api_passphrase'])
    except Exception as e:
        print(f"[ERROR] Failed to decrypt credentials: {e}")
        print("[ERROR] This may indicate corrupted data or wrong ENCRYPTION_KEY")
        return None

    return creds


def calculate_pnl_summary() -> Dict[str, Any]:
    """Calculate P&L summary from orders and fills."""
    conn = get_db()
    cursor = conn.cursor()

    # Get filled orders
    cursor.execute("""
        SELECT COUNT(*) as total_trades,
               SUM(CASE WHEN side = 'sell' THEN filled_avg_price * filled_qty ELSE -filled_avg_price * filled_qty END) as net_pnl
        FROM orders
        WHERE status = 'filled'
    """)
    row = cursor.fetchone()

    total_trades = row['total_trades'] or 0

    # Calculate win rate (simplified - needs proper trade pairing logic)
    cursor.execute("""
        SELECT
            SUM(CASE WHEN side = 'sell' AND filled_avg_price > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN side = 'sell' THEN filled_avg_price * filled_qty ELSE 0 END) as total_wins,
            SUM(CASE WHEN side = 'buy' THEN filled_avg_price * filled_qty ELSE 0 END) as total_losses
        FROM orders
        WHERE status = 'filled'
    """)
    stats = cursor.fetchone()

    winning_trades = stats['winning_trades'] or 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_wins = stats['total_wins'] or 0
    total_losses = stats['total_losses'] or 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

    # Get net P&L from positions
    cursor.execute("SELECT SUM(unrealized_pnl) as total_unrealized FROM positions")
    positions_row = cursor.fetchone()
    unrealized_pnl = positions_row['total_unrealized'] or 0

    conn.close()

    return {
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'net_profit': round((total_wins - total_losses + unrealized_pnl), 2),
        'realized_pnl': round((total_wins - total_losses), 2),
        'unrealized_pnl': round(unrealized_pnl, 2)
    }


# Initialize database on import
init_db()
