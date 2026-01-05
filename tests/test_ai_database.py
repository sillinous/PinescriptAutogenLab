# tests/test_ai_database.py
"""
Tests for AI database operations (ai_database.py)

Tests cover:
- Schema initialization
- TradingView signal storage and retrieval
- AI predictions logging
- ML model metadata storage
- Signal aggregation data
- Feature store operations
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch


@pytest.fixture
def temp_ai_db():
    """Create temporary database for AI tests."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_ai.db')

    # Set environment variable for database path
    with patch.dict(os.environ, {'AI_DATABASE_PATH': db_path}):
        yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    os.rmdir(temp_dir)


class TestSchemaInitialization:
    """Tests for AI database schema initialization."""

    def test_init_ai_schema(self, temp_ai_db):
        """Test that schema initializes correctly."""
        from backend.ai_database import init_ai_schema, get_ai_db_connection

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            # Verify tables exist
            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            expected_tables = {
                'tradingview_signals',
                'ai_predictions',
                'ml_models',
                'signal_aggregation',
                'signal_performance',
                'feature_store',
                'learning_events',
                'market_regimes',
                'strategy_evolution',
                'model_performance'
            }

            for table in expected_tables:
                assert table in tables, f"Table {table} not found in schema"

            conn.close()

    def test_init_schema_idempotent(self, temp_ai_db):
        """Test that schema initialization is idempotent."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            # Initialize twice - should not raise
            init_ai_schema()
            init_ai_schema()

            # Verify no duplicate tables
            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tradingview_signals'")
            count = cursor.fetchone()[0]
            assert count == 1
            conn.close()


class TestTradingViewSignals:
    """Tests for TradingView signal operations."""

    def test_save_tradingview_signal(self, temp_ai_db):
        """Test saving a TradingView signal."""
        from backend.ai_database import init_ai_schema, save_tradingview_signal

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            signal_id = save_tradingview_signal(
                raw_payload={'action': 'buy', 'ticker': 'BTC_USDT'},
                ticker='BTC_USDT',
                action='buy',
                price=50000.0,
                quantity=1.0,
                strategy_name='RSI_Strategy',
                timeframe='1h',
                indicators={'rsi': 25},
                ai_confidence=0.85,
                current_market_price=50100.0
            )

            assert signal_id > 0

    def test_get_similar_signals(self, temp_ai_db):
        """Test retrieving similar signals."""
        from backend.ai_database import init_ai_schema, save_tradingview_signal, get_similar_signals

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            # Save multiple signals
            for i in range(5):
                save_tradingview_signal(
                    raw_payload={'action': 'buy', 'ticker': 'BTC_USDT'},
                    ticker='BTC_USDT',
                    action='buy',
                    price=50000.0 + i * 100,
                    quantity=1.0,
                    strategy_name='RSI_Strategy',
                    timeframe='1h',
                    indicators={'rsi': 25 + i},
                    ai_confidence=0.85
                )

            similar = get_similar_signals('BTC_USDT', 'buy', 'RSI_Strategy', limit=10)

            assert len(similar) == 5

    def test_get_similar_signals_empty(self, temp_ai_db):
        """Test retrieving similar signals when none exist."""
        from backend.ai_database import init_ai_schema, get_similar_signals

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            similar = get_similar_signals('NONEXISTENT', 'buy', 'SomeStrategy')

            assert len(similar) == 0


class TestAIPredictions:
    """Tests for AI prediction logging."""

    def test_log_ai_prediction(self, temp_ai_db):
        """Test logging an AI prediction."""
        from backend.ai_database import init_ai_schema, log_ai_prediction

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            prediction_id = log_ai_prediction(
                model_id=1,
                ticker='BTC_USDT',
                prediction_type='price',
                predicted_value={'direction': 'up', 'magnitude': 0.05},
                confidence=0.82,
                features={'rsi': 30, 'macd': 0.001},
                market_state={'price': 50000, 'volume': 1000000}
            )

            assert prediction_id > 0

    def test_log_prediction_with_actual(self, temp_ai_db):
        """Test logging prediction with actual outcome."""
        from backend.ai_database import init_ai_schema, log_ai_prediction

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            # Log initial prediction
            prediction_id = log_ai_prediction(
                model_id=1,
                ticker='ETH_USDT',
                prediction_type='direction',
                predicted_value={'direction': 'up'},
                confidence=0.75,
                features={},
                market_state={}
            )

            # Update with actual outcome (if function exists)
            # This tests the ability to track prediction accuracy
            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE ai_predictions SET actual_value = ?, is_correct = ? WHERE id = ?",
                ('{"direction": "up"}', 1, prediction_id)
            )
            conn.commit()
            conn.close()

            # Verify update
            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()
            cursor.execute("SELECT is_correct FROM ai_predictions WHERE id = ?", (prediction_id,))
            result = cursor.fetchone()
            assert result[0] == 1
            conn.close()


class TestMLModels:
    """Tests for ML model metadata storage."""

    def test_save_ml_model(self, temp_ai_db):
        """Test saving ML model metadata."""
        from backend.ai_database import init_ai_schema, save_ml_model

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            model_id = save_ml_model(
                name='lstm_btc_v1',
                model_type='LSTM',
                version='1.0.0',
                file_path='/models/lstm_btc_v1.pt',
                hyperparameters={'hidden_size': 128, 'num_layers': 2},
                performance_metrics={'mse': 0.001, 'mae': 0.02}
            )

            assert model_id > 0

    def test_save_model_duplicate_name(self, temp_ai_db):
        """Test saving model with duplicate name (should update or handle)."""
        from backend.ai_database import init_ai_schema, save_ml_model

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            # Save first model
            model_id_1 = save_ml_model(
                name='duplicate_model',
                model_type='LSTM',
                version='1.0.0',
                file_path='/models/model.pt',
                hyperparameters={},
                performance_metrics={}
            )

            # Save with same name - behavior depends on implementation
            model_id_2 = save_ml_model(
                name='duplicate_model',
                model_type='LSTM',
                version='2.0.0',
                file_path='/models/model_v2.pt',
                hyperparameters={},
                performance_metrics={}
            )

            # Both should have valid IDs
            assert model_id_1 > 0
            assert model_id_2 > 0


class TestSignalAggregation:
    """Tests for signal aggregation data storage."""

    def test_save_aggregation_result(self, temp_ai_db):
        """Test saving aggregation results."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            # Direct insert to test table structure
            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO signal_aggregation
                (ticker, aggregated_action, aggregated_confidence, consensus_score,
                 participating_sources, source_signals, final_recommendation, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'BTC_USDT',
                'buy',
                0.85,
                0.9,
                '["tradingview", "ai_model"]',
                '{"tv": 0.8, "ai": 0.9}',
                '{"action": "buy", "size": 0.1}',
                datetime.utcnow().isoformat()
            ))

            conn.commit()
            agg_id = cursor.lastrowid
            conn.close()

            assert agg_id > 0


class TestFeatureStore:
    """Tests for feature store operations."""

    def test_save_features(self, temp_ai_db):
        """Test saving features to feature store."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            # Insert feature data
            cursor.execute("""
                INSERT INTO feature_store
                (ticker, timeframe, feature_set, features_json, computed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                'BTC_USDT',
                '1h',
                'technical_v1',
                '{"rsi_14": 45.5, "macd": 0.002, "bb_upper": 51000}',
                datetime.utcnow().isoformat()
            ))

            conn.commit()
            feature_id = cursor.lastrowid
            conn.close()

            assert feature_id > 0

    def test_get_latest_features(self, temp_ai_db):
        """Test retrieving latest features."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            # Insert multiple feature snapshots
            for i in range(3):
                cursor.execute("""
                    INSERT INTO feature_store
                    (ticker, timeframe, feature_set, features_json, computed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    'ETH_USDT',
                    '1h',
                    'technical_v1',
                    f'{{"rsi_14": {40 + i}}}',
                    (datetime.utcnow() - timedelta(hours=i)).isoformat()
                ))

            conn.commit()

            # Get latest
            cursor.execute("""
                SELECT features_json FROM feature_store
                WHERE ticker = ? AND timeframe = ?
                ORDER BY computed_at DESC LIMIT 1
            """, ('ETH_USDT', '1h'))

            result = cursor.fetchone()
            conn.close()

            assert result is not None
            assert '42' in result[0]  # Latest should have rsi_14: 42


class TestMarketRegimes:
    """Tests for market regime detection storage."""

    def test_save_market_regime(self, temp_ai_db):
        """Test saving detected market regime."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO market_regimes
                (ticker, timeframe, regime_type, confidence, indicators,
                 start_time, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'BTC_USDT',
                '4h',
                'trending_up',
                0.82,
                '{"adx": 35, "trend_strength": "strong"}',
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            ))

            conn.commit()
            regime_id = cursor.lastrowid
            conn.close()

            assert regime_id > 0


class TestPerformanceTracking:
    """Tests for model and signal performance tracking."""

    def test_log_signal_performance(self, temp_ai_db):
        """Test logging signal performance."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO signal_performance
                (signal_id, ticker, action, entry_price, exit_price, pnl, pnl_pct,
                 hold_time_minutes, success, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                1,
                'BTC_USDT',
                'buy',
                50000.0,
                51000.0,
                1000.0,
                2.0,
                120,
                1,
                datetime.utcnow().isoformat()
            ))

            conn.commit()
            perf_id = cursor.lastrowid
            conn.close()

            assert perf_id > 0

    def test_log_model_performance(self, temp_ai_db):
        """Test logging daily model performance."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO model_performance
                (model_id, date, predictions_made, correct_predictions,
                 accuracy, avg_confidence, mse, mae)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                1,
                datetime.utcnow().date().isoformat(),
                100,
                75,
                0.75,
                0.82,
                0.001,
                0.02
            ))

            conn.commit()
            perf_id = cursor.lastrowid
            conn.close()

            assert perf_id > 0


class TestLearningEvents:
    """Tests for learning events and feedback loop."""

    def test_log_learning_event(self, temp_ai_db):
        """Test logging a learning event."""
        from backend.ai_database import init_ai_schema

        with patch('backend.ai_database.AI_DB_PATH', temp_ai_db):
            init_ai_schema()

            conn = sqlite3.connect(temp_ai_db)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO learning_events
                (event_type, model_id, trigger_reason, old_performance,
                 new_performance, changes_made, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'retrain',
                1,
                'performance_degradation',
                '{"accuracy": 0.7}',
                '{"accuracy": 0.85}',
                '{"epochs": 100, "learning_rate": 0.001}',
                datetime.utcnow().isoformat()
            ))

            conn.commit()
            event_id = cursor.lastrowid
            conn.close()

            assert event_id > 0
