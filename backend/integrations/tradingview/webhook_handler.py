# backend/integrations/tradingview/webhook_handler.py

import hashlib
import hmac
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import re
from backend.database import insert_tradingview_signal, update_tradingview_signal_ai_data


class TradingSignal(BaseModel):
    """Structured trading signal from TradingView"""

    signal_id: Optional[int] = Field(None, description="Database ID of the signal")
    ticker: str = Field(..., description="Trading symbol (e.g., AAPL, BTCUSDT)")
    action: str = Field(..., description="Trade action: buy, sell, close")
    price: Optional[float] = Field(None, description="Signal price")
    quantity: Optional[float] = Field(None, description="Order quantity")
    notional: Optional[float] = Field(None, description="Dollar amount")

    # Strategy context
    strategy_name: Optional[str] = Field(None, description="Strategy that generated signal")
    timeframe: Optional[str] = Field(None, description="Chart timeframe (1m, 5m, 1h, etc)")

    # Technical indicators at signal time
    indicators: Dict[str, Any] = Field(default_factory=dict, description="Technical indicator values")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_payload: Optional[Dict[str, Any]] = Field(None, description="Original webhook payload")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Signal confidence score")

    @validator('action')
    def validate_action(cls, v):
        """Ensure action is valid"""
        valid_actions = ['buy', 'sell', 'close', 'long', 'short', 'exit']
        if v.lower() not in valid_actions:
            raise ValueError(f"Invalid action: {v}. Must be one of {valid_actions}")
        return v.lower()

    @validator('ticker')
    def validate_ticker(cls, v):
        """Clean and validate ticker"""
        # Remove any special characters, keep alphanumeric and common symbols
        cleaned = re.sub(r'[^A-Z0-9/\-_.]', '', v.upper())
        if not cleaned:
            raise ValueError("Invalid ticker format")
        return cleaned


class EnrichedSignal(BaseModel):
    """Signal enriched with market context and AI analysis"""

    signal_id: int = Field(..., description="Database ID of the signal")
    signal: TradingSignal

    # Market context
    current_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    volume_24h: Optional[float] = None

    # Technical analysis
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    trend_direction: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    volatility_percentile: Optional[float] = None

    # AI predictions
    ai_confidence: Optional[float] = Field(None, description="AI's confidence in this signal")
    predicted_outcome: Optional[str] = None  # 'profitable', 'unprofitable', 'uncertain'
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Context
    market_regime: Optional[str] = None  # From regime detector
    similar_signals_count: int = 0
    avg_similar_signal_pnl: Optional[float] = None

    enriched_at: datetime = Field(default_factory=datetime.utcnow)


class TradingViewWebhookHandler:
    """
    Handles TradingView webhook alerts and converts them to structured signals.

    TradingView alerts send JSON payloads that we parse, validate, and enrich
    with additional market data and AI analysis.
    """

    def __init__(self, webhook_secret: str = None):
        """
        Initialize webhook handler.

        Args:
            webhook_secret: Secret key for HMAC signature validation
        """
        from backend.config import Config
        self.webhook_secret = webhook_secret or Config.WEBHOOK_SECRET

    def parse_alert(self, payload: Dict[str, Any]) -> TradingSignal:
        """
        Parse TradingView alert payload into structured signal.

        TradingView alert message format examples:

        Simple:
            {"ticker": "AAPL", "action": "buy"}

        With quantity:
            {"ticker": "BTCUSDT", "action": "buy", "quantity": 0.5}

        With indicators:
            {
                "ticker": "{{ticker}}",
                "action": "buy",
                "price": {{close}},
                "indicators": {
                    "rsi": {{rsi}},
                    "macd": {{macd}},
                    "volume": {{volume}}
                },
                "strategy": "RSI Oversold",
                "timeframe": "15m"
            }

        Args:
            payload: Raw webhook payload dict

        Returns:
            TradingSignal object

        Raises:
            ValueError: If payload is invalid
        """
        # Extract core fields
        ticker = payload.get('ticker') or payload.get('symbol')
        action = payload.get('action') or payload.get('side') or payload.get('order')

        if not ticker or not action:
            raise ValueError("Payload must contain 'ticker' and 'action' fields")

        # Build signal
        signal_data = {
            'ticker': ticker,
            'action': action,
            'raw_payload': payload
        }

        # Optional fields
        if 'price' in payload:
            signal_data['price'] = float(payload['price'])

        if 'quantity' in payload or 'qty' in payload:
            signal_data['quantity'] = float(payload.get('quantity') or payload.get('qty'))

        if 'notional' in payload:
            signal_data['notional'] = float(payload['notional'])

        if 'strategy' in payload or 'strategy_name' in payload:
            signal_data['strategy_name'] = payload.get('strategy') or payload.get('strategy_name')

        if 'timeframe' in payload or 'interval' in payload:
            signal_data['timeframe'] = payload.get('timeframe') or payload.get('interval')

        # Technical indicators
        if 'indicators' in payload and isinstance(payload['indicators'], dict):
            signal_data['indicators'] = payload['indicators']
        else:
            # Try to extract common indicator fields
            indicators = {}
            for key in ['rsi', 'macd', 'ema', 'sma', 'bb_upper', 'bb_lower', 'atr', 'volume']:
                if key in payload:
                    indicators[key] = payload[key]
            if indicators:
                signal_data['indicators'] = indicators

        # Create TradingSignal object
        trading_signal = TradingSignal(**signal_data)

        # Store the raw signal in the database and get its ID
        signal_id = insert_tradingview_signal(trading_signal.dict(exclude_none=True))
        trading_signal.signal_id = signal_id

        return trading_signal

    def validate_signature(self, payload_str: str, signature: str) -> bool:
        """
        Validate webhook signature using HMAC-SHA256.

        TradingView doesn't natively support webhook signatures, but you can
        add a signature field to your alert message for security.

        Alert message example:
            {
                "ticker": "AAPL",
                "action": "buy",
                "signature": "computed_hmac_sha256"
            }

        Args:
            payload_str: Raw payload string (before JSON parsing)
            signature: HMAC signature from payload

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.webhook_secret:
            # Security: Reject webhooks if no secret is configured
            import warnings
            warnings.warn(
                "WEBHOOK_SECRET not configured - rejecting webhook for security. "
                "Set WEBHOOK_SECRET environment variable to enable webhooks.",
                UserWarning
            )
            return False

        expected_signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    async def enrich_signal(self, signal: TradingSignal) -> EnrichedSignal:
        """
        Enrich signal with market data and AI analysis.

        This adds:
        - Current market price and spread
        - Support/resistance levels
        - AI confidence score
        - Historical performance of similar signals

        Args:
            signal: Base trading signal

        Returns:
            EnrichedSignal with additional context
        """
        enriched = EnrichedSignal(signal=signal)

        # Get current market data
        try:
            market_data = await self._get_market_data(signal.ticker)
            if market_data:
                enriched.current_price = market_data.get('price')
                enriched.bid = market_data.get('bid')
                enriched.ask = market_data.get('ask')
                if enriched.bid and enriched.ask:
                    enriched.spread = enriched.ask - enriched.bid
                enriched.volume_24h = market_data.get('volume')
        except Exception as e:
            print(f"[WARNING] Failed to fetch market data: {e}")

        # Calculate technical levels (support/resistance)
        try:
            levels = await self._calculate_support_resistance(signal.ticker)
            enriched.support_levels = levels.get('support', [])
            enriched.resistance_levels = levels.get('resistance', [])
        except Exception as e:
            print(f"[WARNING] Failed to calculate S/R levels: {e}")

        # Get AI confidence (if model is trained)
        try:
            ai_analysis = await self._get_ai_prediction(signal)
            enriched.ai_confidence = ai_analysis.get('confidence')
            enriched.predicted_outcome = ai_analysis.get('outcome')
            enriched.risk_score = ai_analysis.get('risk_score')

            # Update the tradingview_signals table with AI data
            if signal.signal_id:
                update_tradingview_signal_ai_data(
                    signal.signal_id,
                    {'outcome': enriched.predicted_outcome, 'risk_score': enriched.risk_score},
                    enriched.ai_confidence
                )

        except Exception as e:
            print(f"[INFO] AI prediction not available: {e}")

        # Get historical performance of similar signals
        try:
            historical = await self._get_similar_signal_performance(signal)
            enriched.similar_signals_count = historical.get('count', 0)
            enriched.avg_similar_signal_pnl = historical.get('avg_pnl')

            # Insert into signal_performance table
            if signal.signal_id and historical.get('pnl') is not None: # Only insert if we have actual performance data
                insert_signal_performance({
                    'signal_id': signal.signal_id,
                    'entry_price': historical.get('entry_price'),
                    'exit_price': historical.get('exit_price'),
                    'pnl': historical.get('pnl'),
                    'duration_minutes': historical.get('duration_minutes'),
                    'success': historical.get('success'),
                    'ai_prediction': {'outcome': enriched.predicted_outcome}, # Use enriched AI data
                    'ai_confidence': enriched.ai_confidence
                })

        except Exception as e:
            print(f"[WARNING] Failed to get historical performance: {e}")

        return enriched

    async def _get_market_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current market data for ticker.

        Uses multiple data sources:
        1. Alpaca (for stocks)
        2. Crypto.com (for crypto)
        3. Fallback APIs
        """
        # Try Crypto.com API first (already integrated)
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                # Use existing Crypto.com endpoint
                response = await client.get(
                    f"http://localhost:8080/price/{ticker}",
                    timeout=5.0
                )
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass

        # Try Alpaca for stocks
        try:
            from backend.brokers.alpaca_client import get_alpaca_client
            alpaca = get_alpaca_client()

            # Get latest quote
            bars = await alpaca.get_bars(ticker, timeframe="1Min", limit=1)
            if bars and 'bars' in bars and bars['bars']:
                latest = bars['bars'][-1]
                return {
                    'price': latest['c'],
                    'volume': latest['v']
                }
        except Exception:
            pass

        return None

    async def _calculate_support_resistance(self, ticker: str) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels using price history.
        """
        from backend.integrations.tradingview.chart_service import get_chart_service
        chart_service = get_chart_service()

        try:
            # Fetch some historical data to calculate S/R levels
            # Using 1D timeframe for broader S/R levels, adjust as needed
            df = await chart_service.get_ohlcv(ticker, timeframe='1D', bars=200)
            if df.empty:
                return {'support': [], 'resistance': []}

            return chart_service.calculate_support_resistance(df)
        except Exception as e:
            print(f"[ERROR] Failed to calculate S/R levels for {ticker}: {e}")
            return {'support': [], 'resistance': []}

    async def _get_ai_prediction(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Get AI model prediction for this signal.

        Returns:
            Dict with 'confidence', 'outcome', 'risk_score'
        """
        # TODO: Integrate with trained AI model (Phase 2)
        # For now, return a neutral prediction
        return {
            'confidence': 0.5,
            'outcome': 'uncertain',
            'risk_score': 0.5
        }

    async def _get_similar_signal_performance(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Query database for performance of similar past signals.

        "Similar" means:
        - Same ticker
        - Same action
        - Same strategy (if specified)
        - Within similar market conditions
        """
        # TODO: Query signal_performance table (Phase 3)
        # For now, return no history
        return {
            'count': 0,
            'avg_pnl': None,
            'pnl': None,
            'entry_price': None,
            'exit_price': None,
            'duration_minutes': None,
            'success': None
        }

    def generate_test_signature(self, payload: Dict[str, Any]) -> str:
        """
        Generate HMAC signature for testing.

        Use this to create signatures for your TradingView alerts.

        Args:
            payload: Alert payload dict

        Returns:
            HMAC-SHA256 signature hex string
        """
        payload_str = json.dumps(payload, sort_keys=True)
        return hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()


# Singleton instance
_webhook_handler: Optional[TradingViewWebhookHandler] = None


def get_webhook_handler() -> TradingViewWebhookHandler:
    """Get or create webhook handler singleton."""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = TradingViewWebhookHandler()
    return _webhook_handler
