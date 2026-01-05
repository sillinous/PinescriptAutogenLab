# backend/ai/features/feature_engineer.py
"""
Automated Feature Engineering Pipeline

Generates 100+ features from raw OHLCV data for ML models:
- Technical indicators (trend, momentum, volatility, volume)
- Statistical features (rolling stats, autocorrelation)
- Pattern-based features (candlestick patterns, support/resistance)
- Time-based features (hour, day of week, etc.)

Also handles feature selection and transformation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest


class FeatureEngineer:
    """
    Automated feature generation and selection for trading ML models.
    """

    def __init__(self):
        """Initialize feature engineer with scalers and selectors."""
        self.scaler = StandardScaler()
        self.selector = None
        self.selected_features: List[str] = []
        self.is_fitted = False

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 100+ additional feature columns
        """
        df = df.copy()

        # Generate each category of features
        df = self.generate_technical_features(df)
        df = self.generate_statistical_features(df)
        df = self.generate_pattern_features(df)
        df = self.generate_time_features(df)
        df = self.generate_price_features(df)

        return df

    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicator features.

        Includes:
        - Trend: SMA, EMA, MACD, ADX
        - Momentum: RSI, Stochastic, CCI, Williams %R
        - Volatility: ATR, Bollinger Bands, Keltner Channels
        - Volume: OBV, MFI, VWAP
        """
        # === TREND INDICATORS ===

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff()

        # Exponential Moving Averages
        for period in [9, 12, 26, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = ((df['macd'] > df['macd_signal']) &
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)

        # === MOMENTUM INDICATORS ===

        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()

        # CCI (Commodity Channel Index)
        for period in [20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

        # Williams %R
        for period in [14]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * ((high_max - df['close']) / (high_max - low_min + 1e-10))

        # === VOLATILITY INDICATORS ===

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        df['atr_21'] = true_range.rolling(window=21).mean()

        # Bollinger Bands
        for period in [20]:
            df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

        # === VOLUME INDICATORS ===

        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # Volume SMA and ratio
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # MFI (Money Flow Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        mf_pos = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
        mf_neg = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum()
        df['mfi_14'] = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-10)))

        # VWAP (Volume Weighted Average Price) - daily reset
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df['vwap'] = (df.groupby('date')
                         .apply(lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum())
                         .reset_index(level=0, drop=True))
            df = df.drop('date', axis=1)

        return df

    def generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical features.

        Includes rolling statistics, autocorrelation, etc.
        """
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Rolling statistics on returns
        for window in [5, 10, 20, 50]:
            returns_window = df['returns'].rolling(window=window)

            df[f'returns_mean_{window}'] = returns_window.mean()
            df[f'returns_std_{window}'] = returns_window.std()
            df[f'returns_skew_{window}'] = returns_window.skew()
            df[f'returns_kurt_{window}'] = returns_window.kurt()

        # Volatility
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)

        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_lag{lag}'] = df['returns'].rolling(window=20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )

        # Z-score of price
        for window in [20, 50]:
            mean = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'zscore_{window}'] = (df['close'] - mean) / (std + 1e-10)

        # High-Low range
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['high_low_range_sma20'] = df['high_low_range'].rolling(window=20).mean()

        return df

    def generate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pattern-based features.

        Includes:
        - Candlestick patterns
        - Support/resistance proximity
        - Trendline breaks
        """
        # Candlestick body and shadow
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close']
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

        # Doji pattern (small body)
        df['is_doji'] = (df['body_pct'] < 0.001).astype(int)

        # Hammer / Hanging Man
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body']) &
                          (df['upper_shadow'] < df['body'])).astype(int)

        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) &
                                   (df['open'].shift(1) > df['close'].shift(1)) &
                                   (df['close'] > df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1))).astype(int)

        df['bearish_engulfing'] = ((df['open'] > df['close']) &
                                   (df['close'].shift(1) > df['open'].shift(1)) &
                                   (df['open'] > df['close'].shift(1)) &
                                   (df['close'] < df['open'].shift(1))).astype(int)

        # Recent highs/lows
        for window in [10, 20, 50]:
            df[f'high_{window}'] = df['high'].rolling(window=window).max()
            df[f'low_{window}'] = df['low'].rolling(window=window).min()
            df[f'dist_to_high_{window}'] = (df[f'high_{window}'] - df['close']) / df['close']
            df[f'dist_to_low_{window}'] = (df['close'] - df[f'low_{window}']) / df['close']

        return df

    def generate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-based features.

        Only works if 'timestamp' column exists.
        """
        if 'timestamp' not in df.columns:
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        # Cyclical encoding (sin/cos for hour and day of week)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Market session (assuming US market hours)
        df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_opening_hour'] = (df['hour'] == 9).astype(int)
        df['is_closing_hour'] = (df['hour'] == 15).astype(int)

        return df

    def generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price-based features.

        Includes momentum, rate of change, etc.
        """
        # Momentum (rate of change)
        for period in [1, 5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['close'].pct_change(periods=period)

        # Price distance from SMAs
        for period in [20, 50, 200]:
            if f'sma_{period}' in df.columns:
                df[f'price_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']

        # Consecutive up/down days
        df['price_change_direction'] = np.sign(df['close'].diff())
        df['consecutive_up'] = (df['price_change_direction']
                               .groupby((df['price_change_direction'] != df['price_change_direction'].shift()).cumsum())
                               .cumsum()
                               .where(df['price_change_direction'] > 0, 0))

        df['consecutive_down'] = (df['price_change_direction']
                                 .groupby((df['price_change_direction'] != df['price_change_direction'].shift()).cumsum())
                                 .cumsum()
                                 .where(df['price_change_direction'] < 0, 0)
                                 .abs())

        return df

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 50,
        method: str = 'mutual_info'
    ) -> List[str]:
        """
        Select top K most important features.

        Args:
            X: Feature DataFrame
            y: Target variable (e.g., future returns)
            k: Number of features to select
            method: Selection method ('mutual_info', 'correlation')

        Returns:
            List of selected feature names
        """
        # Drop non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Handle NaN values
        X_numeric = X_numeric.fillna(0)
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0)

        if method == 'mutual_info':
            # Mutual Information
            mi_scores = mutual_info_regression(X_numeric, y)
            mi_scores = pd.Series(mi_scores, index=X_numeric.columns)
            selected = mi_scores.nlargest(k).index.tolist()

        elif method == 'correlation':
            # Correlation with target
            correlations = X_numeric.corrwith(y).abs()
            selected = correlations.nlargest(k).index.tolist()

        else:
            raise ValueError(f"Unknown method: {method}")

        self.selected_features = selected
        return selected

    def transform(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Transform DataFrame to normalized numpy array for ML.

        Args:
            df: DataFrame with features
            fit: Whether to fit scaler (True for training, False for inference)
            normalize: Whether to normalize features

        Returns:
            Normalized feature array
        """
        # Select features (if already selected)
        if self.selected_features:
            df = df[self.selected_features]

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].copy()

        # Handle NaN and inf
        X = X.fillna(method='ffill').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        if normalize:
            if fit:
                X_scaled = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    raise ValueError("Scaler not fitted. Call with fit=True first.")
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        return X_scaled

    def create_sequences(
        self,
        data: np.ndarray,
        lookback: int = 60,
        forecast_horizon: int = 1
    ) -> tuple:
        """
        Create sequences for time series ML models (LSTM, etc.).

        Args:
            data: 2D array of shape (samples, features)
            lookback: Number of past timesteps to include
            forecast_horizon: Number of future timesteps to predict

        Returns:
            Tuple of (X, y) where:
            - X: 3D array of shape (samples, lookback, features)
            - y: 2D array of shape (samples, forecast_horizon)
        """
        X, y = [], []

        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(data[i:i + forecast_horizon, 0])  # Assuming first column is target

        return np.array(X), np.array(y)


# Singleton instance
_feature_engineer: Optional[FeatureEngineer] = None


def get_feature_engineer() -> FeatureEngineer:
    """Get or create feature engineer singleton."""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()
    return _feature_engineer
