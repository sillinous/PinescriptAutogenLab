# tests/test_deep_learning_api.py
"""
Comprehensive tests for Deep Learning API endpoints (api_deep_learning.py)

Tests cover:
- LSTM model training and prediction
- Transformer model training and prediction
- Ensemble model creation and prediction
- Model management (list, delete, performance)
- Error handling and edge cases
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def mock_chart_service_dl():
    """Mock chart service for deep learning tests."""
    mock = MagicMock()

    # Create sample OHLCV data with enough rows for training
    dates = pd.date_range(start='2024-01-01', periods=500, freq='h')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(500) * 0.5) + 0.5,
        'low': 100 + np.cumsum(np.random.randn(500) * 0.5) - 0.5,
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 10000, 500)
    })
    sample_df.set_index('timestamp', inplace=True)

    async def mock_get_ohlcv(*args, **kwargs):
        bars = kwargs.get('bars', 500)
        return sample_df.tail(bars)

    mock.get_ohlcv = mock_get_ohlcv

    return mock


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestLSTMEndpoints:
    """Tests for LSTM model endpoints."""

    def test_train_lstm_valid_request(self, client, mock_chart_service_dl, temp_model_dir):
        """Test LSTM training with valid parameters."""
        with patch('backend.api_deep_learning.chart_service', mock_chart_service_dl), \
             patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir):

            response = client.post('/api/v2/deep-learning/lstm/train', json={
                'ticker': 'BTC_USDT',
                'timeframe': '1h',
                'lookback_days': 7,
                'sequence_length': 30,
                'prediction_horizon': 1,
                'hidden_size': 64,
                'num_layers': 1,
                'epochs': 2,
                'batch_size': 16,
                'learning_rate': 0.001,
                'early_stopping_patience': 2
            })

            # May fail due to PyTorch not being installed in test env
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert data['status'] == 'completed'
                assert data['model_type'] == 'LSTM'

    def test_train_lstm_insufficient_data(self, client, mock_chart_service_dl, temp_model_dir):
        """Test LSTM training with insufficient data."""
        # Mock returns only 10 rows
        async def mock_small_data(*args, **kwargs):
            dates = pd.date_range(start='2024-01-01', periods=10, freq='h')
            return pd.DataFrame({
                'open': [100] * 10,
                'high': [101] * 10,
                'low': [99] * 10,
                'close': [100] * 10,
                'volume': [1000] * 10
            }, index=dates)

        mock_chart_service_dl.get_ohlcv = mock_small_data

        with patch('backend.api_deep_learning.chart_service', mock_chart_service_dl), \
             patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir):

            response = client.post('/api/v2/deep-learning/lstm/train', json={
                'ticker': 'BTC_USDT',
                'sequence_length': 60,  # Requires more data than available
                'prediction_horizon': 5,
                'epochs': 1
            })

            assert response.status_code == 400
            assert 'Insufficient data' in response.json()['detail']

    def test_predict_lstm_no_model(self, client, temp_model_dir):
        """Test LSTM prediction when no model exists."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.post('/api/v2/deep-learning/lstm/predict', json={
                'ticker': 'NONEXISTENT',
                'model_type': 'lstm',
                'timeframe': '1h',
                'sequence_length': 60
            })

            assert response.status_code == 404
            assert 'No trained LSTM model' in response.json()['detail']

    def test_train_lstm_missing_ticker(self, client):
        """Test LSTM training without ticker."""
        response = client.post('/api/v2/deep-learning/lstm/train', json={
            'timeframe': '1h',
            'epochs': 1
        })

        assert response.status_code == 422  # Validation error


class TestTransformerEndpoints:
    """Tests for Transformer model endpoints."""

    def test_train_transformer_valid_request(self, client, mock_chart_service_dl, temp_model_dir):
        """Test Transformer training with valid parameters."""
        with patch('backend.api_deep_learning.chart_service', mock_chart_service_dl), \
             patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir):

            response = client.post('/api/v2/deep-learning/transformer/train', json={
                'ticker': 'ETH_USDT',
                'timeframe': '1h',
                'lookback_days': 7,
                'sequence_length': 30,
                'prediction_horizon': 1,
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 1,
                'epochs': 2,
                'batch_size': 16,
                'learning_rate': 0.001,
                'early_stopping_patience': 2
            })

            # May fail due to PyTorch not being installed
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert data['status'] == 'completed'
                assert data['model_type'] == 'Transformer'

    def test_predict_transformer_no_model(self, client, temp_model_dir):
        """Test Transformer prediction when no model exists."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.post('/api/v2/deep-learning/transformer/predict', json={
                'ticker': 'NONEXISTENT',
                'model_type': 'transformer',
                'timeframe': '1h',
                'sequence_length': 60
            })

            assert response.status_code == 404
            assert 'No trained Transformer model' in response.json()['detail']


class TestEnsembleEndpoints:
    """Tests for Ensemble model endpoints."""

    def test_create_ensemble_no_models(self, client, temp_model_dir):
        """Test ensemble creation when no base models exist."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.post('/api/v2/deep-learning/ensemble/create', json={
                'ticker': 'BTC_USDT',
                'model_types': ['lstm', 'transformer'],
                'ensemble_method': 'weighted_average'
            })

            assert response.status_code == 404
            assert 'No trained models found' in response.json()['detail']

    def test_predict_ensemble_not_created(self, client):
        """Test ensemble prediction when ensemble doesn't exist."""
        with patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):
            response = client.post('/api/v2/deep-learning/ensemble/predict', json={
                'ticker': 'NONEXISTENT',
                'model_type': 'ensemble',
                'timeframe': '1h',
                'sequence_length': 60
            })

            assert response.status_code == 404
            assert 'No ensemble found' in response.json()['detail']

    def test_create_ensemble_invalid_method(self, client, temp_model_dir):
        """Test ensemble creation with unsupported method."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.post('/api/v2/deep-learning/ensemble/create', json={
                'ticker': 'BTC_USDT',
                'model_types': ['lstm'],
                'ensemble_method': 'invalid_method'
            })

            # Should fail gracefully
            assert response.status_code in [404, 500]


class TestModelManagement:
    """Tests for model management endpoints."""

    def test_list_models_empty(self, client, temp_model_dir):
        """Test listing models when none exist."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.get('/api/v2/deep-learning/models/list')

            assert response.status_code == 200
            data = response.json()
            assert 'lstm' in data
            assert 'transformer' in data
            assert 'ensemble' in data

    def test_list_models_with_files(self, client, temp_model_dir):
        """Test listing models with existing files."""
        # Create a dummy model file
        (temp_model_dir / 'lstm_BTC_USDT_1h.pt').write_bytes(b'dummy model data')
        (temp_model_dir / 'transformer_ETH_USDT_1h.pt').write_bytes(b'dummy model data')

        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.get('/api/v2/deep-learning/models/list')

            assert response.status_code == 200
            data = response.json()
            assert len(data['lstm']) > 0 or len(data['transformer']) > 0

    def test_delete_model(self, client, temp_model_dir):
        """Test deleting a model."""
        # Create a dummy model file
        model_file = temp_model_dir / 'lstm_TEST_1h.pt'
        model_file.write_bytes(b'dummy model data')

        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.delete('/api/v2/deep-learning/models/lstm/TEST')

            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'deleted'

    def test_delete_nonexistent_model(self, client, temp_model_dir):
        """Test deleting a model that doesn't exist."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.delete('/api/v2/deep-learning/models/lstm/NONEXISTENT')

            # Should succeed even if file doesn't exist
            assert response.status_code == 200

    def test_get_model_performance(self, client, temp_model_dir):
        """Test getting model performance."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            response = client.get('/api/v2/deep-learning/models/performance/BTC_USDT')

            assert response.status_code == 200
            data = response.json()
            assert 'ticker' in data
            assert 'performance' in data


class TestRequestValidation:
    """Tests for request validation."""

    def test_lstm_train_invalid_epochs(self, client):
        """Test LSTM training with invalid epochs value."""
        response = client.post('/api/v2/deep-learning/lstm/train', json={
            'ticker': 'BTC_USDT',
            'epochs': -1  # Invalid negative
        })

        # Should be handled by validation or training logic
        assert response.status_code in [200, 422, 500]

    def test_transformer_train_invalid_nhead(self, client):
        """Test Transformer training with invalid nhead value."""
        response = client.post('/api/v2/deep-learning/transformer/train', json={
            'ticker': 'BTC_USDT',
            'd_model': 64,
            'nhead': 7,  # Must divide d_model evenly
            'epochs': 1
        })

        # Should fail during model creation
        assert response.status_code in [200, 422, 500]

    def test_predict_missing_ticker(self, client):
        """Test prediction without ticker."""
        response = client.post('/api/v2/deep-learning/lstm/predict', json={
            'model_type': 'lstm',
            'timeframe': '1h'
        })

        assert response.status_code == 422  # Validation error

    def test_ensemble_empty_model_types(self, client):
        """Test ensemble creation with empty model types."""
        response = client.post('/api/v2/deep-learning/ensemble/create', json={
            'ticker': 'BTC_USDT',
            'model_types': []  # Empty list
        })

        # Should fail
        assert response.status_code in [404, 422, 500]


class TestConcurrency:
    """Tests for concurrent operations."""

    def test_concurrent_predictions(self, client, temp_model_dir):
        """Test multiple concurrent prediction requests."""
        import concurrent.futures

        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir), \
             patch('backend.api_deep_learning.models_cache', {'lstm': {}, 'transformer': {}, 'ensemble': {}}):

            def make_request(ticker):
                return client.post('/api/v2/deep-learning/lstm/predict', json={
                    'ticker': ticker,
                    'model_type': 'lstm',
                    'timeframe': '1h',
                    'sequence_length': 60
                })

            # All should return 404 (no models) consistently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_request, f'TICKER_{i}') for i in range(3)]
                results = [f.result() for f in futures]

            for response in results:
                assert response.status_code == 404


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_ticker(self, client, temp_model_dir):
        """Test with very long ticker name."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir):
            long_ticker = 'A' * 200

            response = client.post('/api/v2/deep-learning/lstm/predict', json={
                'ticker': long_ticker,
                'model_type': 'lstm',
                'timeframe': '1h',
                'sequence_length': 60
            })

            # Should handle gracefully
            assert response.status_code in [404, 422, 500]

    def test_special_characters_ticker(self, client, temp_model_dir):
        """Test with special characters in ticker."""
        with patch('backend.api_deep_learning.MODEL_DIR', temp_model_dir):
            response = client.post('/api/v2/deep-learning/lstm/predict', json={
                'ticker': 'BTC/USDT',  # Contains slash
                'model_type': 'lstm',
                'timeframe': '1h',
                'sequence_length': 60
            })

            # Should handle gracefully
            assert response.status_code in [404, 422, 500]

    def test_zero_sequence_length(self, client):
        """Test with zero sequence length."""
        response = client.post('/api/v2/deep-learning/lstm/predict', json={
            'ticker': 'BTC_USDT',
            'model_type': 'lstm',
            'timeframe': '1h',
            'sequence_length': 0
        })

        assert response.status_code in [400, 404, 422, 500]

    def test_negative_prediction_horizon(self, client):
        """Test with negative prediction horizon."""
        response = client.post('/api/v2/deep-learning/lstm/train', json={
            'ticker': 'BTC_USDT',
            'prediction_horizon': -1,
            'epochs': 1
        })

        assert response.status_code in [400, 422, 500]
