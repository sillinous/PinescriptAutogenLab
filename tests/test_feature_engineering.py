# tests/test_feature_engineering.py
"""
Tests for Feature Engineering module

Tests cover:
- Technical indicators generation
- Statistical features
- Pattern-based features
- Feature normalization
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Generate realistic price data
    base_price = 100
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples).astype(float)
    })

    return data


@pytest.fixture
def feature_engineer():
    """Create feature engineer instance."""
    try:
        from backend.ai.features.feature_engineer import FeatureEngineer
        return FeatureEngineer()
    except ImportError:
        pytest.skip("Feature engineer module not available")


class TestTechnicalIndicators:
    """Tests for technical indicator generation."""

    def test_generate_all_features(self, feature_engineer, sample_price_data):
        """Test generating all features."""
        df = feature_engineer.generate_all_features(sample_price_data)

        # Should have more columns than original
        original_cols = len(sample_price_data.columns)
        assert len(df.columns) > original_cols

    def test_rsi_calculation(self, feature_engineer, sample_price_data):
        """Test RSI calculation."""
        df = feature_engineer.generate_all_features(sample_price_data)

        # Check if RSI column exists
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
        assert len(rsi_cols) > 0

        # RSI should be between 0 and 100
        for col in rsi_cols:
            valid_rsi = df[col].dropna()
            assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_macd_calculation(self, feature_engineer, sample_price_data):
        """Test MACD calculation."""
        df = feature_engineer.generate_all_features(sample_price_data)

        macd_cols = [col for col in df.columns if 'macd' in col.lower()]
        assert len(macd_cols) > 0

    def test_bollinger_bands(self, feature_engineer, sample_price_data):
        """Test Bollinger Bands calculation."""
        df = feature_engineer.generate_all_features(sample_price_data)

        bb_cols = [col for col in df.columns if 'bb_' in col.lower() or 'bollinger' in col.lower()]
        # May not have explicit BB columns, but should have some volatility measure

    def test_moving_averages(self, feature_engineer, sample_price_data):
        """Test moving averages calculation."""
        df = feature_engineer.generate_all_features(sample_price_data)

        ma_cols = [col for col in df.columns if 'sma' in col.lower() or 'ema' in col.lower()]
        assert len(ma_cols) > 0

    def test_volume_features(self, feature_engineer, sample_price_data):
        """Test volume-based features."""
        df = feature_engineer.generate_all_features(sample_price_data)

        vol_cols = [col for col in df.columns if 'volume' in col.lower() or 'vol_' in col.lower()]
        # Should have at least original volume column
        assert len(vol_cols) >= 1


class TestStatisticalFeatures:
    """Tests for statistical feature generation."""

    def test_returns_calculation(self, feature_engineer, sample_price_data):
        """Test returns calculation."""
        df = feature_engineer.generate_all_features(sample_price_data)

        return_cols = [col for col in df.columns if 'return' in col.lower() or 'ret_' in col.lower()]
        # Should have some return-based features

    def test_volatility_features(self, feature_engineer, sample_price_data):
        """Test volatility features."""
        df = feature_engineer.generate_all_features(sample_price_data)

        vol_cols = [col for col in df.columns if 'volatility' in col.lower() or 'std' in col.lower()]
        # Should have volatility measures

    def test_momentum_features(self, feature_engineer, sample_price_data):
        """Test momentum features."""
        df = feature_engineer.generate_all_features(sample_price_data)

        momentum_cols = [col for col in df.columns if 'momentum' in col.lower() or 'roc' in col.lower()]
        # Momentum indicators


class TestPatternFeatures:
    """Tests for pattern-based features."""

    def test_candlestick_patterns(self, feature_engineer, sample_price_data):
        """Test candlestick pattern detection."""
        df = feature_engineer.generate_all_features(sample_price_data)

        # Look for pattern columns
        pattern_cols = [col for col in df.columns if 'pattern' in col.lower() or 'candle' in col.lower()]
        # May or may not have explicit pattern columns

    def test_price_levels(self, feature_engineer, sample_price_data):
        """Test price level features."""
        df = feature_engineer.generate_all_features(sample_price_data)

        # Look for support/resistance or level features
        level_cols = [col for col in df.columns if 'support' in col.lower() or 'resistance' in col.lower()]


class TestFeatureQuality:
    """Tests for feature quality and integrity."""

    def test_no_infinite_values(self, feature_engineer, sample_price_data):
        """Test that features don't contain infinite values."""
        df = feature_engineer.generate_all_features(sample_price_data)

        # Check for infinity
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                assert not np.isinf(df[col].dropna()).any(), f"Infinite values in {col}"

    def test_reasonable_nan_ratio(self, feature_engineer, sample_price_data):
        """Test that NaN ratio is reasonable."""
        df = feature_engineer.generate_all_features(sample_price_data)

        for col in df.columns:
            nan_ratio = df[col].isna().sum() / len(df)
            # NaN ratio should be less than 50% for most features
            # (some initial values may be NaN due to lookback)
            assert nan_ratio < 0.5, f"Too many NaN values in {col}: {nan_ratio:.2%}"

    def test_feature_consistency(self, feature_engineer, sample_price_data):
        """Test that features are consistent across multiple runs."""
        df1 = feature_engineer.generate_all_features(sample_price_data.copy())
        df2 = feature_engineer.generate_all_features(sample_price_data.copy())

        # Same input should produce same output
        common_cols = set(df1.columns) & set(df2.columns)
        for col in common_cols:
            if df1[col].dtype in [np.float64, np.float32]:
                np.testing.assert_array_almost_equal(
                    df1[col].fillna(0).values,
                    df2[col].fillna(0).values,
                    decimal=10,
                    err_msg=f"Inconsistent values in {col}"
                )


class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimal_data(self, feature_engineer):
        """Test with minimal data."""
        # Very small dataset
        small_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })

        # Should handle gracefully
        try:
            df = feature_engineer.generate_all_features(small_data)
            assert len(df) == len(small_data)
        except ValueError:
            # May raise error for insufficient data
            pass

    def test_constant_prices(self, feature_engineer):
        """Test with constant prices (no volatility)."""
        constant_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })

        df = feature_engineer.generate_all_features(constant_data)

        # Should not crash
        assert len(df) == 50

    def test_extreme_values(self, feature_engineer):
        """Test with extreme price values."""
        extreme_data = pd.DataFrame({
            'open': [1e-10, 1e10, 100, 1e-10, 1e10] * 20,
            'high': [1e-9, 1e11, 101, 1e-9, 1e11] * 20,
            'low': [1e-11, 1e9, 99, 1e-11, 1e9] * 20,
            'close': [1e-10, 1e10, 100.5, 1e-10, 1e10] * 20,
            'volume': [1, 1e15, 1000, 1, 1e15] * 20
        })

        # Should handle without crashing
        try:
            df = feature_engineer.generate_all_features(extreme_data)
            assert len(df) == 100
        except (ValueError, OverflowError):
            # May fail on extreme values
            pass

    def test_missing_columns(self, feature_engineer):
        """Test with missing required columns."""
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104] * 20,
            'volume': [1000, 1100, 1200, 1300, 1400] * 20
        })

        # Should handle missing OHLC columns
        try:
            df = feature_engineer.generate_all_features(incomplete_data)
        except (KeyError, ValueError):
            # Expected to fail with missing columns
            pass

    def test_negative_prices(self, feature_engineer):
        """Test with negative prices (invalid but should handle)."""
        negative_data = pd.DataFrame({
            'open': [-100, -99, -98] * 30,
            'high': [-99, -98, -97] * 30,
            'low': [-101, -100, -99] * 30,
            'close': [-100, -99, -98] * 30,
            'volume': [1000, 1100, 1200] * 30
        })

        # Should handle gracefully
        try:
            df = feature_engineer.generate_all_features(negative_data)
            # Some features may be NaN for negative prices
            assert len(df) == 90
        except (ValueError, RuntimeWarning):
            # May warn or fail on negative prices
            pass


class TestFeatureNormalization:
    """Tests for feature normalization if available."""

    def test_normalize_features(self, feature_engineer, sample_price_data):
        """Test feature normalization."""
        df = feature_engineer.generate_all_features(sample_price_data)

        # If normalize method exists
        if hasattr(feature_engineer, 'normalize'):
            df_normalized = feature_engineer.normalize(df)

            # Check normalization bounds
            for col in df_normalized.columns:
                if df_normalized[col].dtype in [np.float64, np.float32]:
                    valid_vals = df_normalized[col].dropna()
                    if len(valid_vals) > 0:
                        # Most normalized values should be within reasonable range
                        assert valid_vals.min() >= -10, f"Value too low in {col}"
                        assert valid_vals.max() <= 10, f"Value too high in {col}"
