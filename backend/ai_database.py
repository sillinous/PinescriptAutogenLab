# backend/ai_database.py
"""
AI/ML specific database schema and operations.

This module extends the base database with tables for:
- ML models registry
- AI predictions logging
- TradingView signals
- Signal performance tracking
- Feature store
- Learning events
- Market regimes
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from backend.database import get_db, DB_PATH


def init_ai_schema():
    """
    Initialize AI/ML database schema.

    Creates tables for storing AI models, predictions, signals, and learning data.
    """
    conn = get_db()
    cursor = conn.cursor()

    # ML Models Registry
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,  -- 'RL', 'LSTM', 'CNN', 'XGBoost', 'ensemble'
            version TEXT NOT NULL,
            file_path TEXT,  -- Path to saved model file
            hyperparameters TEXT,  -- JSON string
            performance_metrics TEXT,  -- JSON: sharpe, win_rate, etc.
            training_data_info TEXT,  -- JSON: date range, symbols, etc.
            trained_at TIMESTAMP,
            deployed_at TIMESTAMP,
            status TEXT DEFAULT 'training',  -- 'training', 'deployed', 'archived', 'failed'
            created_by TEXT DEFAULT 'system',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, version)
        )
    """)

    # AI Predictions Log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            ticker TEXT NOT NULL,
            prediction_type TEXT NOT NULL,  -- 'action', 'price', 'pattern', 'regime'
            predicted_value TEXT,  -- JSON: action, confidence, etc.
            confidence REAL,
            features TEXT,  -- JSON: feature values used
            market_state TEXT,  -- JSON: market conditions at prediction time
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Feedback (filled later for learning)
            actual_outcome TEXT,  -- JSON: what actually happened
            outcome_timestamp TIMESTAMP,
            accuracy_score REAL,  -- How accurate was prediction
            pnl REAL,  -- P&L if this was a trading signal

            FOREIGN KEY (model_id) REFERENCES ml_models(id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON ai_predictions(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ai_predictions(created_at)")

    # TradingView Signals
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tradingview_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_payload TEXT NOT NULL,  -- Original webhook JSON
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,  -- 'buy', 'sell', 'close'
            price REAL,
            quantity REAL,
            notional REAL,

            -- Strategy context
            strategy_name TEXT,
            timeframe TEXT,
            indicators TEXT,  -- JSON: RSI, MACD, etc. at signal time

            -- Enrichment
            current_market_price REAL,
            spread REAL,
            support_levels TEXT,  -- JSON array
            resistance_levels TEXT,  -- JSON array
            ai_confidence REAL,  -- AI's confidence in this signal
            ai_prediction TEXT,  -- JSON: AI's analysis

            -- Execution
            order_id INTEGER,  -- Link to orders table if executed
            executed BOOLEAN DEFAULT 0,

            -- Performance tracking
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            duration_minutes INTEGER,
            success BOOLEAN,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            executed_at TIMESTAMP,
            closed_at TIMESTAMP,

            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tv_signals_ticker ON tradingview_signals(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tv_signals_strategy ON tradingview_signals(strategy_name)")

    # Signal Performance (for learning)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            signal_type TEXT,  -- 'tradingview', 'ai_generated', 'manual'

            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL,

            pnl REAL,
            pnl_percent REAL,
            duration_minutes INTEGER,

            -- Context at entry
            market_regime TEXT,  -- From regime detector
            volatility_percentile REAL,
            rsi REAL,
            macd REAL,
            trend_direction TEXT,

            -- Outcome
            success BOOLEAN,
            win_amount REAL,
            loss_amount REAL,

            entry_timestamp TIMESTAMP NOT NULL,
            exit_timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (signal_id) REFERENCES tradingview_signals(id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_perf_ticker ON signal_performance(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_perf_success ON signal_performance(success)")

    # Feature Store (for ML training)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            features TEXT NOT NULL,  -- JSON: all calculated features

            -- Quick access to common features
            close_price REAL,
            volume REAL,
            rsi REAL,
            macd REAL,
            volatility REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(ticker, timestamp)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_ticker_time ON feature_store(ticker, timestamp)")

    # Learning Events (feedback loop)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,  -- 'trade_closed', 'model_retrained', 'strategy_evolved', 'regime_change'

            trade_id INTEGER,
            model_id INTEGER,
            signal_id INTEGER,

            insights TEXT,  -- JSON: what was learned
            actions_taken TEXT,  -- JSON: what actions were triggered

            -- Metrics
            previous_performance TEXT,  -- JSON: before learning
            new_performance TEXT,  -- JSON: after learning
            improvement_score REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (trade_id) REFERENCES orders(id),
            FOREIGN KEY (model_id) REFERENCES ml_models(id),
            FOREIGN KEY (signal_id) REFERENCES tradingview_signals(id)
        )
    """)

    # Market Regimes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_regimes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            regime_type TEXT NOT NULL,  -- 'bull_trending', 'bear_trending', 'range_bound', 'high_volatility'
            confidence REAL NOT NULL,

            -- Regime characteristics
            avg_volatility REAL,
            trend_strength REAL,
            detected_by TEXT,  -- Model/algorithm that detected it

            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_minutes INTEGER,

            -- Associated strategy
            recommended_strategy TEXT,  -- Which strategy works best in this regime

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_regimes_ticker ON market_regimes(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_regimes_active ON market_regimes(ticker, end_time)")

    # Strategy Evolution (genetic algorithm results)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            population_size INTEGER,

            -- Best strategy from this generation
            strategy_genes TEXT NOT NULL,  -- JSON: parameters, rules, etc.
            fitness_score REAL NOT NULL,

            -- Performance
            backtest_sharpe REAL,
            backtest_win_rate REAL,
            backtest_profit_factor REAL,
            backtest_max_drawdown REAL,

            -- Parentage
            parent_1_id INTEGER,
            parent_2_id INTEGER,
            mutation_applied BOOLEAN,

            deployed BOOLEAN DEFAULT 0,
            deployed_at TIMESTAMP,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (parent_1_id) REFERENCES strategy_evolution(id),
            FOREIGN KEY (parent_2_id) REFERENCES strategy_evolution(id)
        )
    """)

    # Model Performance Tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            evaluation_date DATE NOT NULL,

            -- Prediction accuracy
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,

            -- Trading performance
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            win_rate REAL,
            total_pnl REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            profit_factor REAL,

            -- Model health
            prediction_latency_ms REAL,
            feature_drift_score REAL,
            model_confidence_avg REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (model_id) REFERENCES ml_models(id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_model ON model_performance(model_id, evaluation_date)")

    conn.commit()
    conn.close()

    print("[INFO] AI database schema initialized successfully")


# Helper functions for AI operations

def save_ml_model(
    name: str,
    model_type: str,
    version: str,
    file_path: str,
    hyperparameters: Dict[str, Any],
    performance_metrics: Dict[str, Any]
) -> int:
    """Save ML model metadata to database."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO ml_models
        (name, model_type, version, file_path, hyperparameters, performance_metrics, trained_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'trained')
    """, (
        name,
        model_type,
        version,
        file_path,
        json.dumps(hyperparameters),
        json.dumps(performance_metrics),
        datetime.utcnow()
    ))

    model_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return model_id


def log_ai_prediction(
    model_id: int,
    ticker: str,
    prediction_type: str,
    predicted_value: Dict[str, Any],
    confidence: float,
    features: Dict[str, Any],
    market_state: Dict[str, Any]
) -> int:
    """Log an AI prediction."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO ai_predictions
        (model_id, ticker, prediction_type, predicted_value, confidence, features, market_state)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        model_id,
        ticker,
        prediction_type,
        json.dumps(predicted_value),
        confidence,
        json.dumps(features),
        json.dumps(market_state)
    ))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return prediction_id


def save_tradingview_signal(
    raw_payload: Dict[str, Any],
    ticker: str,
    action: str,
    **kwargs
) -> int:
    """Save TradingView signal to database."""
    conn = get_db()
    cursor = conn.cursor()

    fields = ['raw_payload', 'ticker', 'action']
    values = [json.dumps(raw_payload), ticker, action]

    # Add optional fields
    for key, value in kwargs.items():
        if value is not None:
            fields.append(key)
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value))
            else:
                values.append(value)

    placeholders = ','.join(['?'] * len(values))
    field_names = ','.join(fields)

    cursor.execute(f"""
        INSERT INTO tradingview_signals ({field_names})
        VALUES ({placeholders})
    """, values)

    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return signal_id


def update_signal_performance(
    signal_id: int,
    exit_price: float,
    pnl: float,
    success: bool
):
    """Update signal with performance data after trade closes."""
    conn = get_db()
    cursor = conn.cursor()

    # Get signal entry data
    cursor.execute("SELECT created_at, price FROM tradingview_signals WHERE id = ?", (signal_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return

    entry_time = datetime.fromisoformat(row['created_at'])
    entry_price = row['price']

    duration = int((datetime.utcnow() - entry_time).total_seconds() / 60)

    cursor.execute("""
        UPDATE tradingview_signals
        SET exit_price = ?,
            pnl = ?,
            success = ?,
            duration_minutes = ?,
            closed_at = ?
        WHERE id = ?
    """, (exit_price, pnl, success, duration, datetime.utcnow(), signal_id))

    conn.commit()
    conn.close()


def get_similar_signals(
    ticker: str,
    action: str,
    strategy_name: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get historical performance of similar signals."""
    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT * FROM tradingview_signals
        WHERE ticker = ?
        AND action = ?
        AND success IS NOT NULL
    """
    params = [ticker, action]

    if strategy_name:
        query += " AND strategy_name = ?"
        params.append(strategy_name)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def save_feature_snapshot(
    ticker: str,
    timestamp: datetime,
    features: Dict[str, Any]
):
    """Save feature snapshot for ML training."""
    conn = get_db()
    cursor = conn.cursor()

    # Extract common features for quick access
    close_price = features.get('close')
    volume = features.get('volume')
    rsi = features.get('rsi')
    macd = features.get('macd')
    volatility = features.get('volatility_20')

    cursor.execute("""
        INSERT OR REPLACE INTO feature_store
        (ticker, timestamp, features, close_price, volume, rsi, macd, volatility)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        timestamp,
        json.dumps(features),
        close_price,
        volume,
        rsi,
        macd,
        volatility
    ))

    conn.commit()
    conn.close()


def log_learning_event(
    event_type: str,
    insights: Dict[str, Any],
    actions_taken: Dict[str, Any],
    **kwargs
):
    """Log a learning event."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO learning_events
        (event_type, insights, actions_taken, trade_id, model_id, signal_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        event_type,
        json.dumps(insights),
        json.dumps(actions_taken),
        kwargs.get('trade_id'),
        kwargs.get('model_id'),
        kwargs.get('signal_id')
    ))

    conn.commit()
    conn.close()


def save_market_regime(
    ticker: str,
    regime_type: str,
    confidence: float,
    recommended_strategy: Optional[str] = None,
    **kwargs
) -> int:
    """Save detected market regime."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO market_regimes
        (ticker, regime_type, confidence, recommended_strategy, start_time, avg_volatility, trend_strength, detected_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        regime_type,
        confidence,
        recommended_strategy,
        datetime.utcnow(),
        kwargs.get('avg_volatility'),
        kwargs.get('trend_strength'),
        kwargs.get('detected_by', 'system')
    ))

    regime_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return regime_id


def get_active_regime(ticker: str) -> Optional[Dict[str, Any]]:
    """Get currently active market regime for ticker."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM market_regimes
        WHERE ticker = ?
        AND end_time IS NULL
        ORDER BY start_time DESC
        LIMIT 1
    """, (ticker,))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


# Initialize schema on import
if __name__ == "__main__":
    init_ai_schema()
