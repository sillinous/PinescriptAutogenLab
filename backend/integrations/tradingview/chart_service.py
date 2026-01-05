# backend/integrations/tradingview/chart_service.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import httpx
from backend.config import Config


class ChartDataService:
    """
    Fetches and processes chart data for AI analysis.

    Supports multiple data sources with automatic fallback:
    1. Alpaca (stocks/ETFs)
    2. Crypto.com (crypto)
    3. Yahoo Finance (backup)
    """

    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = timedelta(minutes=1)  # 1 minute cache

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        bars: int = 500,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTCUSDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1D')
            bars: Number of bars to fetch
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If no data available from any source
        """
        cache_key = f"{symbol}_{timeframe}_{bars}"

        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                return cached_data.copy()

        # Try data sources in order
        df = None

        # 1. Try Crypto.com (for crypto)
        if any(quote in symbol.upper() for quote in ['USDT', 'USDC', 'USD', 'BTC', 'ETH']):
            try:
                df = await self._fetch_from_crypto_com(symbol, timeframe, bars)
            except Exception as e:
                print(f"[INFO] Crypto.com failed: {e}")

        # 2. Try Alpaca (for stocks)
        if df is None:
            try:
                df = await self._fetch_from_alpaca(symbol, timeframe, bars)
            except Exception as e:
                print(f"[INFO] Alpaca failed: {e}")

        # 3. Try Yahoo Finance (backup)
        if df is None:
            try:
                df = await self._fetch_from_yahoo(symbol, timeframe, bars)
            except Exception as e:
                print(f"[INFO] Yahoo Finance failed: {e}")

        if df is None or df.empty:
            raise ValueError(f"Failed to fetch data for {symbol} from all sources")

        # Standardize DataFrame
        df = self._standardize_dataframe(df)

        # Cache result
        self.cache[cache_key] = (df.copy(), datetime.utcnow())

        return df

    async def _fetch_from_crypto_com(
        self,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Crypto.com API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:8080/candles/{symbol}",
                params={'interval': timeframe, 'limit': bars},  # Backend /candles endpoint uses 'limit' parameter
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            if 'candles' not in data or not data['candles']:
                return None

            candles = data['candles']
            df = pd.DataFrame(candles)

            # Rename columns to standard format
            df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }, inplace=True)

            return df

    async def _fetch_from_alpaca(
        self,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpaca"""
        from backend.brokers.alpaca_client import get_alpaca_client

        # Map timeframe to Alpaca format
        timeframe_map = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1Hour',
            '4h': '4Hour',
            '1d': '1Day',
            '1D': '1Day'
        }
        alpaca_tf = timeframe_map.get(timeframe.lower(), '1Hour')

        alpaca = get_alpaca_client()
        data = await alpaca.get_bars(symbol, timeframe=alpaca_tf, limit=bars)

        if not data or 'bars' not in data or not data['bars']:
            return None

        bars_data = data['bars']
        df = pd.DataFrame(bars_data)

        # Alpaca uses: t (timestamp), o, h, l, c, v
        df.rename(columns={'t': 'timestamp'}, inplace=True)

        return df

    async def _fetch_from_yahoo(
        self,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf

            # Normalize symbol for Yahoo Finance
            # Convert BTC_USDT -> BTC-USD, ETH_USDT -> ETH-USD, etc.
            yahoo_symbol = symbol.replace('_', '-')
            if yahoo_symbol.endswith('-USDT'):
                yahoo_symbol = yahoo_symbol.replace('-USDT', '-USD')
            elif yahoo_symbol.endswith('-USDC'):
                yahoo_symbol = yahoo_symbol.replace('-USDC', '-USD')

            # Map timeframe to Yahoo format
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '1d': '1d',
                '1D': '1d'
            }
            interval = interval_map.get(timeframe.lower(), '1h')

            # Determine period based on interval and Yahoo Finance limitations
            # Yahoo Finance data availability:
            # - 1m: max 7 days
            # - 5m: max 60 days
            # - 15m, 30m, 1h: max 730 days (2 years)
            # - 1d: unlimited
            if interval == '1m':
                # 1-minute data limited to 7 days
                period = '7d'
            elif interval == '5m':
                # 5-minute data limited to 60 days
                if bars <= 500:
                    period = '5d'
                else:
                    period = '60d'
            elif interval in ['15m', '30m']:
                # 15/30-minute data available for up to 60 days
                if bars <= 100:
                    period = '5d'
                elif bars <= 500:
                    period = '1mo'
                else:
                    period = '60d'
            elif interval == '1h':
                # Hourly data available for up to 2 years
                if bars <= 100:
                    period = '5d'
                elif bars <= 500:
                    period = '1mo'
                elif bars <= 1500:
                    period = '3mo'
                else:
                    period = '2y'
            else:  # 1d
                # Daily data - use max available
                if bars <= 100:
                    period = '6mo'
                elif bars <= 500:
                    period = '2y'
                else:
                    period = 'max'

            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            # Yahoo uses: Open, High, Low, Close, Volume
            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            # Take last N bars
            df = df.tail(bars)

            return df

        except ImportError:
            print("[WARNING] yfinance not installed. Install with: pip install yfinance")
            return None

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format.

        Ensures:
        - Columns: timestamp, open, high, low, close, volume
        - Timestamp is datetime
        - Sorted by timestamp ascending
        - No missing values
        """
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if pd.api.types.is_integer_dtype(df['timestamp']):
                # Unix timestamp (seconds)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Convert OHLCV to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        # Keep only required columns
        df = df[required_cols]

        return df

    async def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Adds 50+ indicators to the DataFrame:
        - Trend: SMA, EMA, MACD, ADX
        - Momentum: RSI, Stochastic, CCI, Williams %R
        - Volatility: ATR, Bollinger Bands, Keltner Channels
        - Volume: OBV, MFI, VWAP

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        df = df.copy()

        # Simple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in [12, 26, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # Volume moving average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(periods=period)

        # Volatility (rolling std of returns)
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()

        return df

    def prepare_for_ml(self, df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
        """
        Prepare data for ML model input.

        Converts DataFrame to normalized numpy array suitable for:
        - LSTM/GRU time series models
        - CNN chart image models
        - RL environment observations

        Args:
            df: DataFrame with OHLCV and indicators
            lookback: Number of historical bars to include in each sample

        Returns:
            3D numpy array of shape (samples, lookback, features)
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].copy()

        # Drop timestamp if present
        if 'timestamp' in data.columns:
            data = data.drop('timestamp', axis=1)

        # Fill NaN values (from indicator calculation)
        data = data.fillna(method='bfill').fillna(0)

        # Normalize each feature to [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)

        # Create sequences
        sequences = []
        for i in range(lookback, len(data_normalized)):
            sequences.append(data_normalized[i - lookback:i])

        return np.array(sequences)

    async def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol"""
        df = await self.get_ohlcv(symbol, timeframe='1m', bars=1)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        return float(df.iloc[-1]['close'])

    def calculate_support_resistance(
        self,
        df: pd.DataFrame,
        num_levels: int = 3
    ) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels using pivot points.

        Args:
            df: DataFrame with OHLCV data
            num_levels: Number of S/R levels to calculate

        Returns:
            Dict with 'support' and 'resistance' keys
        """
        # Use recent data (last 100 bars)
        recent = df.tail(100).copy()

        # Find local maxima (resistance) and minima (support)
        from scipy.signal import argrelextrema

        # Local maxima (resistance levels)
        local_max_idx = argrelextrema(recent['high'].values, np.greater, order=5)[0]
        resistance_prices = recent.iloc[local_max_idx]['high'].values

        # Local minima (support levels)
        local_min_idx = argrelextrema(recent['low'].values, np.less, order=5)[0]
        support_prices = recent.iloc[local_min_idx]['low'].values

        # Cluster nearby levels
        def cluster_levels(prices, tolerance=0.02):
            """Group levels within tolerance%"""
            if len(prices) == 0:
                return []

            prices = sorted(prices, reverse=True)
            clusters = []
            current_cluster = [prices[0]]

            for price in prices[1:]:
                if abs(price - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(price)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [price]

            if current_cluster:
                clusters.append(np.mean(current_cluster))

            return clusters

        resistance = cluster_levels(resistance_prices)[:num_levels]
        support = cluster_levels(support_prices)[:num_levels]

        return {
            'resistance': sorted(resistance, reverse=True),
            'support': sorted(support, reverse=True)
        }


# Singleton instance
_chart_service: Optional[ChartDataService] = None


def get_chart_service() -> ChartDataService:
    """Get or create chart service singleton."""
    global _chart_service
    if _chart_service is None:
        _chart_service = ChartDataService()
    return _chart_service
