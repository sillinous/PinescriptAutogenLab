# tests/test_ai_database.py
"""
Tests for AI database operations, aligned with the schema in backend/database.py
"""

import pytest
import sqlite3
from datetime import datetime
from backend.database import get_db
from backend.ai_database import (
    init_ai_schema,
    save_ml_model,
    log_ai_prediction,
    save_tradingview_signal,
    update_signal_performance,
    get_similar_signals,
    save_feature_snapshot,
    log_learning_event,
    save_market_regime,
    get_active_regime
)
# This import is also needed for the test
from backend.database import update_tradingview_signal_ai_data


class TestSchemaInitialization:
    """Tests for AI database schema initialization."""

    def test_ai_tables_exist(self, db):
        """Test that AI-related tables from database.py are created."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_ai_tables = {
            'ml_models',
            'ai_predictions',
            'tradingview_signals',
            'signal_performance',
            'feature_store',
            'learning_events',
            'market_regimes',
            'strategy_evolution',
            'model_performance',
        }

        for table in expected_ai_tables:
            assert table in tables, f"AI-related table '{table}' not found in schema"

class TestTradingViewSignalOperations:
    """Tests for TradingView signal database operations."""

    def test_insert_tradingview_signal(self, db):
        """Test saving a TradingView signal using the correct schema."""
        signal_id = save_tradingview_signal(
            raw_payload={'text': 'BTCUSDT buy'},
            ticker='BTC_USDT',
            action='buy',
            price=50000.0,
            strategy_name='RSI_Strategy',
            timeframe='1h',
            indicators={'rsi': 25}
        )
        assert signal_id > 0

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tradingview_signals WHERE id = ?", (signal_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row['ticker'] == 'BTC_USDT'
        assert row['action'] == 'buy'
        assert row['strategy_name'] == 'RSI_Strategy'

    def test_update_tradingview_signal_ai_data(self, db):
        """Test updating a signal with AI analysis data."""
        # This function is in database.py, not ai_database.py
        from backend.database import insert_tradingview_signal

        signal_id = insert_tradingview_signal({
            'raw_payload': '{"text": "ETHUSDT sell"}',
            'ticker': 'ETH_USDT',
            'action': 'sell',
            'price': 4000.0
        })

        ai_prediction_data = {'reason': 'overbought', 'confidence_breakdown': [0.8, 0.9]}
        ai_confidence_score = 0.85

        update_tradingview_signal_ai_data(signal_id, ai_prediction_data, ai_confidence_score)

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT ai_prediction, ai_confidence FROM tradingview_signals WHERE id = ?", (signal_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row['ai_confidence'] == ai_confidence_score
        assert '"reason": "overbought"' in row['ai_prediction']

class TestMLModelOperations:
    """Tests for ML model metadata storage."""

    def test_save_ml_model(self, db):
        """Test saving ML model metadata with the correct schema."""
        model_id = save_ml_model(
            name='lstm_btc_v1',
            model_type='LSTM', # Corresponds to 'type' column in db
            version='1.0.0',
            file_path='/models/lstm_btc_v1.pt',
            hyperparameters={'hidden_size': 128},
            performance_metrics={'mse': 0.001}
        )
        assert model_id > 0

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ml_models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row['name'] == 'lstm_btc_v1'
        assert row['type'] == 'LSTM' # Verify 'type' column
        assert '"hidden_size": 128' in row['hyperparameters']

    def test_duplicate_model_version_fails(self, db):
        """Test that saving a model with the same name and version raises an error."""
        save_ml_model('unique_model', 'XGBoost', '1.0', '/path/m1', {}, {})
        with pytest.raises(sqlite3.IntegrityError):
            # This should fail because of the UNIQUE(name, version) constraint
            save_ml_model('unique_model', 'XGBoost', '1.0', '/path/m2', {}, {})

class TestAIPredictionLogging:
    """Tests for logging AI predictions."""

    def test_log_ai_prediction(self, db):
        """Test logging a prediction."""
        prediction_id = log_ai_prediction(
            model_id=1,
            ticker='BTC_USDT',
            prediction_type='action',
            predicted_value={'action': 'buy'},
            confidence=0.9,
            features={},
            market_state={}
        )
        assert prediction_id > 0
