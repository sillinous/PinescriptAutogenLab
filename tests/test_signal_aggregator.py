# tests/test_signal_aggregator.py
"""
Tests for Signal Aggregator module
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock
from backend.ai.signal_aggregator import SignalAggregator, SignalSource, get_signal_aggregator
from backend.integrations.tradingview.webhook_handler import TradingSignal

@pytest.fixture
def signal_aggregator() -> SignalAggregator:
    """Create a fresh signal aggregator instance for each test."""
    # Ensure singleton is reset for testing
    from backend.ai import signal_aggregator as sa_module
    sa_module._signal_aggregator = None
    return get_signal_aggregator()

@pytest.fixture
def sample_signal_sources() -> list[SignalSource]:
    """Create valid sample signal sources for testing."""
    return [
        SignalSource(
            source_type='tradingview',
            source_name='RSI_Oversold',
            signal=TradingSignal(ticker='BTC_USDT', action='buy', price=50000.0),
            confidence=0.8,
            metadata={'rsi': 25}
        ),
        SignalSource(
            source_type='ai_agent',
            source_name='lstm_v1',
            signal=TradingSignal(ticker='BTC_USDT', action='buy'),
            confidence=0.75,
            metadata={'prediction': 51000}
        ),
        SignalSource(
            source_type='technical_indicators',
            source_name='MACD_crossover',
            signal=TradingSignal(ticker='BTC_USDT', action='buy'),
            confidence=0.7,
            metadata={'signal': 'bullish_crossover'}
        )
    ]

class TestSignalSourceCreation:
    """Tests for signal source creation helper methods."""

    @pytest.mark.asyncio
    async def test_create_tradingview_signal(self, signal_aggregator):
        """Test creating signal from an Enriched TradingView signal."""
        mock_enriched_signal = MagicMock()
        mock_enriched_signal.signal = TradingSignal(ticker='ETH_USDT', action='sell', strategy_name='EMA_Cross')
        mock_enriched_signal.ai_confidence = 0.82
        mock_enriched_signal.support_levels = [3000]
        mock_enriched_signal.resistance_levels = [3200]
        mock_enriched_signal.trend_direction = 'down'

        signal_source = await signal_aggregator.create_signal_from_tradingview(mock_enriched_signal)

        assert signal_source.source_type == 'tradingview'
        assert signal_source.source_name == 'EMA_Cross'
        assert signal_source.signal.ticker == 'ETH_USDT'
        assert signal_source.signal.action == 'sell'
        assert signal_source.confidence == 0.82

    @pytest.mark.asyncio
    async def test_create_ai_signal(self, signal_aggregator):
        """Test creating a signal from an AI model's output."""
        signal_source = await signal_aggregator.create_signal_from_ai(
            ticker='BTC_USDT',
            action='buy',
            confidence=0.85,
            model_name='lstm_predictor',
            reasoning={'prediction': 52000, 'current': 50000}
        )
        assert signal_source.source_type == 'ai_agent'
        assert signal_source.source_name == 'lstm_predictor'
        assert signal_source.signal.ticker == 'BTC_USDT'
        assert signal_source.signal.action == 'buy'
        assert signal_source.confidence == 0.85

class TestSignalAggregation:
    """Tests for the core signal aggregation logic."""

    def test_aggregate_multiple_signals(self, signal_aggregator, sample_signal_sources):
        """Test aggregating multiple harmonious signals."""
        result = signal_aggregator.aggregate(sample_signal_sources)
        assert result.ticker == 'BTC_USDT'
        assert result.action == 'buy'
        assert result.aggregated_confidence > 0.7
        assert result.consensus_score == 1.0 # All signals agree on action

    def test_aggregate_conflicting_signals(self, signal_aggregator):
        """Test aggregating conflicting buy/sell signals."""
        conflicting_sources = [
            SignalSource(source_type='A', source_name='s1', signal=TradingSignal(ticker='BTC_USDT', action='buy'), confidence=0.9),
            SignalSource(source_type='B', source_name='s2', signal=TradingSignal(ticker='BTC_USDT', action='sell'), confidence=0.8),
        ]
        result = signal_aggregator.aggregate(conflicting_sources)
        assert result.action == 'buy' # Higher confidence wins
        assert result.consensus_score < 1.0

    def test_aggregate_single_signal(self, signal_aggregator):
        """Test aggregating a single signal."""
        single_source = [SignalSource(source_type='A', source_name='s1', signal=TradingSignal(ticker='ETH_USDT', action='sell'), confidence=0.75)]
        result = signal_aggregator.aggregate(single_source)
        assert result.ticker == 'ETH_USDT'
        assert result.action == 'sell'
        assert result.aggregated_confidence == 0.75
        assert result.consensus_score == 1.0

    def test_aggregate_empty_signals_raises_error(self, signal_aggregator):
        """Test that aggregating an empty list raises a ValueError."""
        with pytest.raises(ValueError):
            signal_aggregator.aggregate([])

class TestExecutionDecision:
    """Tests for the execution decision logic."""

    def test_should_execute_high_confidence(self, signal_aggregator, sample_signal_sources):
        """Test execution decision with high confidence and consensus."""
        agg_signal = signal_aggregator.aggregate(sample_signal_sources)
        assert signal_aggregator.should_execute_signal(agg_signal, min_confidence=0.6, min_consensus=0.5) is True

    def test_should_not_execute_low_confidence(self, signal_aggregator):
        """Test execution decision with low confidence."""
        low_conf_sources = [SignalSource(source_type='A', source_name='s1', signal=TradingSignal(ticker='BTC_USDT', action='buy'), confidence=0.3)]
        agg_signal = signal_aggregator.aggregate(low_conf_sources)
        assert signal_aggregator.should_execute_signal(agg_signal, min_confidence=0.6) is False

    def test_should_not_execute_low_consensus(self, signal_aggregator):
        """Test execution decision with low consensus."""
        conflicting_sources = [
            SignalSource(source_type='A', source_name='s1', signal=TradingSignal(ticker='BTC_USDT', action='buy'), confidence=0.8),
            SignalSource(source_type='B', source_name='s2', signal=TradingSignal(ticker='BTC_USDT', action='sell'), confidence=0.8),
        ]
        agg_signal = signal_aggregator.aggregate(conflicting_sources)
        # Consensus will be low
        assert signal_aggregator.should_execute_signal(agg_signal, min_consensus=0.7) is False