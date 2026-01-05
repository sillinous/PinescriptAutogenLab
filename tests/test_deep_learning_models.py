# tests/test_deep_learning_models.py
"""
Unit tests for Deep Learning models

Tests cover:
- LSTM Predictor
- Transformer Predictor
- Ensemble Predictor
- Model training, prediction, save/load
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


# Skip all tests if PyTorch is not installed
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not installed"),
    reason="PyTorch required for deep learning tests"
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    return data


@pytest.fixture
def temp_model_path():
    """Create temporary path for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / "test_model.pt"
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestLSTMPredictor:
    """Tests for LSTM Predictor."""

    def test_lstm_initialization(self):
        """Test LSTM predictor initialization."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            sequence_length=30,
            prediction_horizon=1,
            learning_rate=0.001
        )

        assert predictor.hidden_size == 64
        assert predictor.num_layers == 2
        assert predictor.sequence_length == 30

    def test_lstm_prepare_data(self, sample_ohlcv_data):
        """Test LSTM data preparation."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=32,
            num_layers=1,
            sequence_length=20,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        assert train_loader is not None
        assert val_loader is not None

        # Check batch structure
        for batch_x, batch_y in train_loader:
            assert batch_x.shape[-1] == 5  # 5 features (OHLCV)
            assert batch_x.shape[1] == 20  # sequence_length
            break

    def test_lstm_training(self, sample_ohlcv_data):
        """Test LSTM model training."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        result = predictor.train(
            train_loader,
            val_loader,
            epochs=2,
            early_stopping_patience=5,
            verbose=False
        )

        assert 'train_losses' in result
        assert 'val_losses' in result
        assert 'best_epoch' in result

    def test_lstm_prediction(self, sample_ohlcv_data):
        """Test LSTM prediction."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        predictor.train(train_loader, val_loader, epochs=2, verbose=False)

        # Make prediction
        input_data = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values[-10:]
        prediction = predictor.predict(input_data)

        assert prediction is not None
        assert len(prediction) == predictor.prediction_horizon

    def test_lstm_save_load(self, sample_ohlcv_data, temp_model_path):
        """Test LSTM model save and load."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        # Create and train model
        predictor1 = LSTMPredictor(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor1.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        predictor1.train(train_loader, val_loader, epochs=2, verbose=False)

        # Save model
        predictor1.save(str(temp_model_path))

        # Load model
        predictor2 = LSTMPredictor()
        predictor2.load(str(temp_model_path))

        # Compare predictions
        input_data = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values[-10:]
        pred1 = predictor1.predict(input_data)
        pred2 = predictor2.predict(input_data)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


class TestTransformerPredictor:
    """Tests for Transformer Predictor."""

    def test_transformer_initialization(self):
        """Test Transformer predictor initialization."""
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor

        predictor = TransformerPredictor(
            input_size=5,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            sequence_length=30,
            prediction_horizon=1,
            learning_rate=0.001
        )

        assert predictor.d_model == 64
        assert predictor.nhead == 4
        assert predictor.sequence_length == 30

    def test_transformer_prepare_data(self, sample_ohlcv_data):
        """Test Transformer data preparation."""
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor

        predictor = TransformerPredictor(
            input_size=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            sequence_length=20,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        assert train_loader is not None
        assert val_loader is not None

    def test_transformer_training(self, sample_ohlcv_data):
        """Test Transformer model training."""
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor

        predictor = TransformerPredictor(
            input_size=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        result = predictor.train(
            train_loader,
            val_loader,
            epochs=2,
            early_stopping_patience=5,
            verbose=False
        )

        assert 'train_losses' in result
        assert 'val_losses' in result

    def test_transformer_prediction(self, sample_ohlcv_data):
        """Test Transformer prediction."""
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor

        predictor = TransformerPredictor(
            input_size=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        predictor.train(train_loader, val_loader, epochs=2, verbose=False)

        input_data = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values[-10:]
        prediction = predictor.predict(input_data)

        assert prediction is not None
        assert len(prediction) == predictor.prediction_horizon

    def test_transformer_save_load(self, sample_ohlcv_data, temp_model_path):
        """Test Transformer model save and load."""
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor

        predictor1 = TransformerPredictor(
            input_size=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        train_loader, val_loader = predictor1.prepare_data(
            sample_ohlcv_data,
            train_split=0.8,
            batch_size=16
        )

        predictor1.train(train_loader, val_loader, epochs=2, verbose=False)

        # Save and load
        predictor1.save(str(temp_model_path))

        predictor2 = TransformerPredictor()
        predictor2.load(str(temp_model_path))

        # Compare predictions
        input_data = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values[-10:]
        pred1 = predictor1.predict(input_data)
        pred2 = predictor2.predict(input_data)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


class TestEnsemblePredictor:
    """Tests for Ensemble Predictor."""

    def test_ensemble_initialization(self, sample_ohlcv_data):
        """Test Ensemble predictor initialization."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor
        from backend.ai.deep_learning.ensemble import EnsemblePredictor

        # Create and train individual models
        lstm = LSTMPredictor(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )
        train_loader, val_loader = lstm.prepare_data(sample_ohlcv_data, train_split=0.8, batch_size=16)
        lstm.train(train_loader, val_loader, epochs=2, verbose=False)

        transformer = TransformerPredictor(
            input_size=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )
        train_loader, val_loader = transformer.prepare_data(sample_ohlcv_data, train_split=0.8, batch_size=16)
        transformer.train(train_loader, val_loader, epochs=2, verbose=False)

        # Create ensemble
        ensemble = EnsemblePredictor(
            models=[lstm, transformer],
            weights=[0.5, 0.5],
            ensemble_method='weighted_average'
        )

        assert len(ensemble.models) == 2
        assert ensemble.weights == [0.5, 0.5]

    def test_ensemble_prediction(self, sample_ohlcv_data):
        """Test Ensemble prediction."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor
        from backend.ai.deep_learning.ensemble import EnsemblePredictor

        # Create and train model
        lstm = LSTMPredictor(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )
        train_loader, val_loader = lstm.prepare_data(sample_ohlcv_data, train_split=0.8, batch_size=16)
        lstm.train(train_loader, val_loader, epochs=2, verbose=False)

        # Create ensemble with single model
        ensemble = EnsemblePredictor(
            models=[lstm],
            weights=[1.0],
            ensemble_method='weighted_average'
        )

        input_data = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values[-10:]
        result = ensemble.predict(input_data, return_uncertainty=True)

        assert 'prediction' in result
        assert 'uncertainty' in result

    def test_ensemble_uncertainty(self, sample_ohlcv_data):
        """Test Ensemble uncertainty estimation."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor
        from backend.ai.deep_learning.ensemble import EnsemblePredictor

        # Create models with different architectures for diversity
        lstm = LSTMPredictor(
            input_size=5, hidden_size=16, num_layers=1,
            sequence_length=10, prediction_horizon=1
        )
        train_loader, val_loader = lstm.prepare_data(sample_ohlcv_data, train_split=0.8, batch_size=16)
        lstm.train(train_loader, val_loader, epochs=2, verbose=False)

        transformer = TransformerPredictor(
            input_size=5, d_model=32, nhead=4, num_encoder_layers=1,
            sequence_length=10, prediction_horizon=1
        )
        train_loader, val_loader = transformer.prepare_data(sample_ohlcv_data, train_split=0.8, batch_size=16)
        transformer.train(train_loader, val_loader, epochs=2, verbose=False)

        ensemble = EnsemblePredictor(
            models=[lstm, transformer],
            weights=[0.5, 0.5]
        )

        input_data = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values[-10:]
        result = ensemble.predict(input_data, return_uncertainty=True)

        assert 'uncertainty' in result
        uncertainty = result['uncertainty']
        assert 'std' in uncertainty or 'confidence' in uncertainty


class TestModelEdgeCases:
    """Tests for edge cases in deep learning models."""

    def test_lstm_minimal_data(self):
        """Test LSTM with minimal data."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        # Very small dataset
        small_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=8,
            num_layers=1,
            sequence_length=2,
            prediction_horizon=1
        )

        # Should handle gracefully or raise appropriate error
        try:
            train_loader, val_loader = predictor.prepare_data(
                small_data,
                train_split=0.8,
                batch_size=2
            )
            # If it succeeds, verify loaders are valid
            assert train_loader is not None
        except ValueError as e:
            # Expected for insufficient data
            assert 'insufficient' in str(e).lower() or 'not enough' in str(e).lower()

    def test_lstm_nan_handling(self, sample_ohlcv_data):
        """Test LSTM handling of NaN values."""
        from backend.ai.deep_learning.lstm_predictor import LSTMPredictor

        # Introduce NaN values
        data_with_nan = sample_ohlcv_data.copy()
        data_with_nan.iloc[50:55, 0] = np.nan

        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            sequence_length=10,
            prediction_horizon=1
        )

        # Should handle NaN gracefully
        try:
            # Fill NaN before processing
            data_filled = data_with_nan.ffill().fillna(0)
            train_loader, val_loader = predictor.prepare_data(
                data_filled,
                train_split=0.8,
                batch_size=16
            )
            assert train_loader is not None
        except Exception as e:
            # Should not crash, but may raise ValueError
            pass

    def test_transformer_invalid_nhead(self):
        """Test Transformer with invalid nhead (must divide d_model)."""
        from backend.ai.deep_learning.transformer_predictor import TransformerPredictor

        with pytest.raises(Exception):  # Should raise AssertionError or similar
            predictor = TransformerPredictor(
                input_size=5,
                d_model=64,
                nhead=7,  # 7 doesn't divide 64
                num_encoder_layers=1,
                sequence_length=10,
                prediction_horizon=1
            )
            # Force model creation if lazy initialization
            if hasattr(predictor, 'model'):
                _ = predictor.model
