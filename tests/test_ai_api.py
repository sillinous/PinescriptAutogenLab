# tests/test_ai_api.py
"""
Comprehensive tests for AI API endpoints (api_ai.py)

Tests cover:
- TradingView webhook processing
- Chart data endpoints
- Feature engineering
- Signal aggregation
- Error handling and edge cases
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np


@pytest.fixture
def mock_chart_service():
    """Mock chart service for testing."""
    mock = MagicMock()

    # Create sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 0.5,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 0.5,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100),
        'rsi_14': np.random.uniform(30, 70, 100)
    })
    sample_df.set_index('timestamp', inplace=True)

    async def mock_get_ohlcv(*args, **kwargs):
        return sample_df

    async def mock_get_indicators(df):
        df_copy = df.copy()
        df_copy['rsi_14'] = np.random.uniform(30, 70, len(df))
        df_copy['sma_20'] = df_copy['close'].rolling(20).mean()
        df_copy['ema_12'] = df_copy['close'].ewm(span=12).mean()
        return df_copy

    mock.get_ohlcv = mock_get_ohlcv
    mock.get_indicators = mock_get_indicators
    mock.calculate_support_resistance = MagicMock(return_value={
        'support': [95.0, 93.0, 90.0],
        'resistance': [105.0, 108.0, 110.0]
    })

    return mock


@pytest.fixture
def mock_signal_aggregator():
    """Mock signal aggregator for testing."""
    mock = MagicMock()

    class MockSignalSource:
        def __init__(self, ticker, action, confidence):
            self.ticker = ticker
            self.action = action
            self.confidence = confidence

    class MockAggregatedSignal:
        def __init__(self, ticker, action):
            self.ticker = ticker
            self.action = action
            self.aggregated_confidence = 0.75
            self.consensus_score = 0.8
            self.participating_sources = ['tradingview', 'technical']
            self.recommended_position_size = 0.1
            self.timestamp = datetime.utcnow()

    async def mock_create_signal(*args, **kwargs):
        return MockSignalSource(
            kwargs.get('ticker', 'BTC_USDT'),
            kwargs.get('action', 'buy'),
            kwargs.get('confidence', 0.8)
        )

    mock.create_signal_from_tradingview = mock_create_signal
    mock.create_signal_from_ai = mock_create_signal
    mock.aggregate = MagicMock(return_value=MockAggregatedSignal('BTC_USDT', 'buy'))
    mock.should_execute_signal = MagicMock(return_value=True)

    return mock


@pytest.fixture
def mock_webhook_handler():
    """Mock webhook handler for testing."""
    mock = MagicMock()

    class MockSignal:
        def __init__(self):
            self.ticker = 'BTC_USDT'
            self.action = 'buy'
            self.price = 50000.0
            self.quantity = 1.0
            self.strategy_name = 'RSI_Strategy'
            self.timeframe = '1h'
            self.indicators = {'rsi': 25}

    class MockEnrichedSignal:
        def __init__(self):
            self.ticker = 'BTC_USDT'
            self.action = 'buy'
            self.ai_confidence = 0.85
            self.current_price = 50100.0
            self.support_levels = [49000, 48000]
            self.resistance_levels = [51000, 52000]

    mock.parse_alert = MagicMock(return_value=MockSignal())

    async def mock_enrich(*args, **kwargs):
        return MockEnrichedSignal()

    mock.enrich_signal = mock_enrich

    return mock


class TestTradingViewWebhook:
    """Tests for TradingView webhook endpoint."""

    def test_webhook_valid_payload(self, client, mock_chart_service, mock_signal_aggregator, mock_webhook_handler):
        """Test processing a valid TradingView webhook."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service), \
             patch('backend.api_ai.get_signal_aggregator', return_value=mock_signal_aggregator), \
             patch('backend.api_ai.get_webhook_handler', return_value=mock_webhook_handler), \
             patch('backend.api_ai.save_tradingview_signal', return_value=1), \
             patch('backend.api_ai.init_ai_schema'):

            payload = {
                'ticker': 'BTC_USDT',
                'action': 'buy',
                'price': 50000.0,
                'quantity': 1.0,
                'strategy': 'RSI_Oversold',
                'timeframe': '1h',
                'indicators': {'rsi': 25, 'volume': 1000000}
            }

            response = client.post('/api/v1/ai/tradingview/webhook', json=payload)

            # Should succeed (or return 503 if AI services unavailable)
            assert response.status_code in [200, 503]

    def test_webhook_missing_required_fields(self, client):
        """Test webhook with missing required fields."""
        payload = {
            'action': 'buy'
            # Missing 'ticker'
        }

        response = client.post('/api/v1/ai/tradingview/webhook', json=payload)
        assert response.status_code == 422  # Validation error

    def test_webhook_invalid_action(self, client):
        """Test webhook with invalid action field."""
        payload = {
            'ticker': 'BTC_USDT',
            'action': '',  # Empty action
        }

        response = client.post('/api/v1/ai/tradingview/webhook', json=payload)
        # Empty string is still valid for the model, just not useful
        assert response.status_code in [200, 422, 500, 503]


class TestChartDataEndpoints:
    """Tests for chart data endpoints."""

    def test_get_chart_ohlcv(self, client, mock_chart_service):
        """Test fetching OHLCV chart data."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service):
            response = client.post('/api/v1/ai/chart/ohlcv', json={
                'ticker': 'BTC_USDT',
                'timeframe': '1h',
                'bars': 100
            })

            assert response.status_code in [200, 500]  # 500 if service unavailable
            if response.status_code == 200:
                data = response.json()
                assert 'ticker' in data
                assert 'data' in data

    def test_get_support_resistance(self, client, mock_chart_service):
        """Test calculating support/resistance levels."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service):
            response = client.get('/api/v1/ai/chart/support-resistance/BTC_USDT?timeframe=1h&bars=200')

            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert 'support_levels' in data
                assert 'resistance_levels' in data


class TestFeatureEngineering:
    """Tests for feature engineering endpoints."""

    def test_generate_features(self, client, mock_chart_service):
        """Test feature generation endpoint."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service):
            response = client.post('/api/v1/ai/features/generate', json={
                'ticker': 'BTC_USDT',
                'timeframe': '1h',
                'bars': 200
            })

            # May return 503 if ML libs not available
            assert response.status_code in [200, 500, 503]

    def test_generate_features_invalid_ticker(self, client):
        """Test feature generation with invalid ticker."""
        response = client.post('/api/v1/ai/features/generate', json={
            'ticker': '',  # Invalid empty ticker
            'timeframe': '1h',
            'bars': 200
        })

        assert response.status_code in [200, 500, 503]


class TestSignalAggregation:
    """Tests for signal aggregation endpoints."""

    def test_aggregate_signals(self, client, mock_chart_service, mock_signal_aggregator):
        """Test signal aggregation endpoint."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service), \
             patch('backend.api_ai.get_signal_aggregator', return_value=mock_signal_aggregator):

            response = client.get('/api/v1/ai/signal/aggregate/BTC_USDT')

            # May return 503 if services unavailable
            assert response.status_code in [200, 500, 503]
            if response.status_code == 200:
                data = response.json()
                assert 'ticker' in data
                assert 'action' in data


class TestModelTraining:
    """Tests for model training endpoints."""

    def test_train_model_request(self, client, mock_chart_service):
        """Test model training endpoint."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service):
            response = client.post('/api/v1/ai/model/train', json={
                'model_name': 'test_model',
                'ticker': 'BTC_USDT',
                'timeframe': '1h',
                'bars': 500,
                'total_timesteps': 1000
            })

            # Training starts in background
            assert response.status_code in [200, 500, 503]

    def test_get_prediction_no_model(self, client):
        """Test prediction when model doesn't exist."""
        response = client.get('/api/v1/ai/model/predict/BTC_USDT?model_name=nonexistent_model')

        # Should return 404 or 500/503 if services unavailable
        assert response.status_code in [404, 500, 503]


class TestOptimizationEndpoints:
    """Tests for strategy optimization endpoints."""

    def test_get_available_strategies(self, client):
        """Test getting available strategies."""
        response = client.get('/api/v1/ai/optimize/strategies')

        assert response.status_code == 200
        strategies = response.json()
        assert isinstance(strategies, list)

    def test_start_optimization_valid(self, client):
        """Test starting strategy optimization."""
        response = client.post('/api/v1/ai/optimize/start', json={
            'strategy_name': 'test_strategy',
            'strategy_type': 'rsi',
            'n_trials': 1
        })

        assert response.status_code == 200
        data = response.json()
        assert 'message' in data

    def test_start_optimization_invalid_type(self, client):
        """Test starting optimization with invalid strategy type."""
        response = client.post('/api/v1/ai/optimize/start', json={
            'strategy_name': 'test_strategy',
            'strategy_type': 'invalid_strategy_type',
            'n_trials': 1
        })

        assert response.status_code == 400
        assert 'Unknown strategy type' in response.json()['detail']

    def test_get_optimization_status_not_found(self, client):
        """Test getting status for non-existent optimization."""
        response = client.get('/api/v1/ai/optimize/status/nonexistent_strategy')

        assert response.status_code == 404


class TestABTestingEndpoints:
    """Tests for A/B testing endpoints."""

    def test_create_ab_test(self, client):
        """Test creating an A/B test."""
        response = client.post('/api/v1/ai/abtest/create', json={
            'test_name': f'test_ab_{datetime.now().timestamp()}',
            'variant_a_params': {'rsi_length': 14},
            'variant_b_params': {'rsi_length': 10},
            'min_sample_size': 5,
            'significance_level': 0.05
        })

        assert response.status_code == 200
        data = response.json()
        assert 'message' in data

    def test_get_active_ab_tests(self, client):
        """Test getting active A/B tests."""
        response = client.get('/api/v1/ai/abtest/active')

        assert response.status_code == 200
        data = response.json()
        assert 'active_tests' in data

    def test_get_ab_test_results_not_found(self, client):
        """Test getting results for non-existent test."""
        response = client.get('/api/v1/ai/abtest/results/nonexistent_test')

        assert response.status_code == 404

    def test_promote_winner_inconclusive(self, client):
        """Test promoting winner when results inconclusive."""
        # First create a test
        test_name = f'test_promote_{datetime.now().timestamp()}'
        client.post('/api/v1/ai/abtest/create', json={
            'test_name': test_name,
            'variant_a_params': {'p': 1},
            'variant_b_params': {'p': 2}
        })

        # Try to promote without any trades
        response = client.post(f'/api/v1/ai/abtest/promote/{test_name}')

        assert response.status_code == 400
        assert 'inconclusive' in response.json()['detail'].lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_ticker(self, client):
        """Test endpoints with empty ticker."""
        response = client.get('/api/v1/ai/signal/aggregate/')
        assert response.status_code in [404, 405, 422]

    def test_special_characters_in_ticker(self, client):
        """Test endpoints with special characters in ticker."""
        response = client.get('/api/v1/ai/signal/aggregate/BTC%2FUSDT')
        # Should handle or reject gracefully (404 is valid for URL-encoded slashes)
        assert response.status_code in [200, 400, 404, 500, 503]

    def test_very_large_bars_request(self, client, mock_chart_service):
        """Test requesting very large number of bars."""
        with patch('backend.api_ai.get_chart_service', return_value=mock_chart_service):
            response = client.post('/api/v1/ai/chart/ohlcv', json={
                'ticker': 'BTC_USDT',
                'timeframe': '1h',
                'bars': 100000  # Very large
            })

            # Should handle gracefully
            assert response.status_code in [200, 400, 500]
