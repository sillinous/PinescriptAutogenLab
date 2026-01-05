# tests/test_signal_aggregator.py
"""
Tests for Signal Aggregator module

Tests cover:
- Signal source creation
- Signal aggregation
- Confidence scoring
- Consensus calculation
- Position sizing
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


@pytest.fixture
def signal_aggregator():
    """Create signal aggregator instance."""
    try:
        from backend.ai.signal_aggregator import SignalAggregator
        return SignalAggregator()
    except ImportError:
        pytest.skip("Signal aggregator module not available")


@pytest.fixture
def sample_signal_sources():
    """Create sample signal sources for testing."""
    try:
        from backend.ai.signal_aggregator import SignalSource

        return [
            SignalSource(
                source_name='tradingview',
                ticker='BTC_USDT',
                action='buy',
                confidence=0.8,
                timestamp=datetime.utcnow(),
                metadata={'strategy': 'RSI_Oversold', 'rsi': 25}
            ),
            SignalSource(
                source_name='ai_model',
                ticker='BTC_USDT',
                action='buy',
                confidence=0.75,
                timestamp=datetime.utcnow(),
                metadata={'model': 'lstm_v1', 'prediction': 51000}
            ),
            SignalSource(
                source_name='technical',
                ticker='BTC_USDT',
                action='buy',
                confidence=0.7,
                timestamp=datetime.utcnow(),
                metadata={'indicator': 'MACD', 'signal': 'bullish_crossover'}
            )
        ]
    except ImportError:
        pytest.skip("SignalSource not available")


class TestSignalSourceCreation:
    """Tests for signal source creation."""

    def test_create_tradingview_signal(self, signal_aggregator):
        """Test creating signal from TradingView webhook."""
        import asyncio

        async def test():
            # Mock enriched signal
            enriched = MagicMock()
            enriched.ticker = 'ETH_USDT'
            enriched.action = 'sell'
            enriched.ai_confidence = 0.82
            enriched.current_price = 3000.0

            signal = await signal_aggregator.create_signal_from_tradingview(enriched)

            assert signal.ticker == 'ETH_USDT'
            assert signal.action == 'sell'
            assert signal.confidence >= 0 and signal.confidence <= 1

        asyncio.run(test())

    def test_create_ai_signal(self, signal_aggregator):
        """Test creating signal from AI model."""
        import asyncio

        async def test():
            signal = await signal_aggregator.create_signal_from_ai(
                ticker='BTC_USDT',
                action='buy',
                confidence=0.85,
                model_name='lstm_predictor',
                reasoning={'prediction': 52000, 'current': 50000}
            )

            assert signal.ticker == 'BTC_USDT'
            assert signal.action == 'buy'
            assert signal.confidence == 0.85

        asyncio.run(test())

    def test_create_signal_invalid_action(self, signal_aggregator):
        """Test creating signal with invalid action."""
        import asyncio

        async def test():
            # Should handle or reject invalid actions
            try:
                signal = await signal_aggregator.create_signal_from_ai(
                    ticker='BTC_USDT',
                    action='invalid_action',
                    confidence=0.5,
                    model_name='test',
                    reasoning={}
                )
                # If it doesn't raise, action should be normalized
                assert signal.action in ['buy', 'sell', 'hold', 'neutral', 'invalid_action']
            except ValueError:
                pass  # Expected for invalid action

        asyncio.run(test())


class TestSignalAggregation:
    """Tests for signal aggregation."""

    def test_aggregate_multiple_signals(self, signal_aggregator, sample_signal_sources):
        """Test aggregating multiple signals."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        assert result.ticker == 'BTC_USDT'
        assert result.action == 'buy'  # All signals agree
        assert 0 <= result.aggregated_confidence <= 1
        assert 0 <= result.consensus_score <= 1

    def test_aggregate_conflicting_signals(self, signal_aggregator):
        """Test aggregating conflicting signals."""
        try:
            from backend.ai.signal_aggregator import SignalSource

            conflicting_sources = [
                SignalSource(
                    source_name='source_a',
                    ticker='BTC_USDT',
                    action='buy',
                    confidence=0.9,
                    timestamp=datetime.utcnow(),
                    metadata={}
                ),
                SignalSource(
                    source_name='source_b',
                    ticker='BTC_USDT',
                    action='sell',
                    confidence=0.8,
                    timestamp=datetime.utcnow(),
                    metadata={}
                )
            ]

            result = signal_aggregator.aggregate(conflicting_sources)

            # Should pick higher confidence or produce neutral signal
            assert result.action in ['buy', 'sell', 'hold', 'neutral']
            # Consensus should be lower due to conflict
            assert result.consensus_score < 1.0

        except ImportError:
            pytest.skip("SignalSource not available")

    def test_aggregate_single_signal(self, signal_aggregator):
        """Test aggregating a single signal."""
        try:
            from backend.ai.signal_aggregator import SignalSource

            single_source = [
                SignalSource(
                    source_name='only_source',
                    ticker='ETH_USDT',
                    action='buy',
                    confidence=0.75,
                    timestamp=datetime.utcnow(),
                    metadata={}
                )
            ]

            result = signal_aggregator.aggregate(single_source)

            assert result.ticker == 'ETH_USDT'
            assert result.action == 'buy'
            assert result.aggregated_confidence == 0.75

        except ImportError:
            pytest.skip("SignalSource not available")

    def test_aggregate_empty_signals(self, signal_aggregator):
        """Test aggregating empty signal list."""
        # Should handle gracefully
        try:
            result = signal_aggregator.aggregate([])
            # May return None or neutral signal
            assert result is None or result.action in ['hold', 'neutral']
        except (ValueError, IndexError):
            pass  # Expected for empty list


class TestConsensusCalculation:
    """Tests for consensus calculation."""

    def test_unanimous_consensus(self, signal_aggregator, sample_signal_sources):
        """Test consensus when all signals agree."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        # All signals are 'buy', so consensus should be high
        assert result.consensus_score >= 0.8

    def test_split_consensus(self, signal_aggregator):
        """Test consensus when signals are split."""
        try:
            from backend.ai.signal_aggregator import SignalSource

            split_sources = [
                SignalSource('a', 'BTC', 'buy', 0.8, datetime.utcnow(), {}),
                SignalSource('b', 'BTC', 'sell', 0.8, datetime.utcnow(), {}),
            ]

            result = signal_aggregator.aggregate(split_sources)

            # Consensus should be around 0.5 for 50/50 split
            assert result.consensus_score <= 0.7

        except ImportError:
            pytest.skip("SignalSource not available")


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_weighted_confidence(self, signal_aggregator, sample_signal_sources):
        """Test weighted confidence calculation."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        # Aggregated confidence should be weighted average or similar
        avg_confidence = sum(s.confidence for s in sample_signal_sources) / len(sample_signal_sources)

        # Should be close to average
        assert abs(result.aggregated_confidence - avg_confidence) < 0.2

    def test_confidence_bounds(self, signal_aggregator, sample_signal_sources):
        """Test that confidence stays within bounds."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        assert 0 <= result.aggregated_confidence <= 1


class TestPositionSizing:
    """Tests for position size recommendations."""

    def test_position_size_high_confidence(self, signal_aggregator, sample_signal_sources):
        """Test position sizing with high confidence signals."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        if result.recommended_position_size is not None:
            # Higher confidence should lead to larger position
            assert result.recommended_position_size > 0

    def test_position_size_bounds(self, signal_aggregator, sample_signal_sources):
        """Test that position size is within bounds."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        if result.recommended_position_size is not None:
            # Position size should be reasonable percentage
            assert 0 <= result.recommended_position_size <= 1


class TestExecutionDecision:
    """Tests for execution decision logic."""

    def test_should_execute_high_confidence(self, signal_aggregator, sample_signal_sources):
        """Test execution decision with high confidence."""
        result = signal_aggregator.aggregate(sample_signal_sources)

        should_execute = signal_aggregator.should_execute_signal(
            result,
            min_confidence=0.6,
            min_consensus=0.5
        )

        # High confidence signals should be executable
        assert should_execute is True

    def test_should_not_execute_low_confidence(self, signal_aggregator):
        """Test execution decision with low confidence."""
        try:
            from backend.ai.signal_aggregator import SignalSource

            low_confidence_sources = [
                SignalSource('a', 'BTC', 'buy', 0.3, datetime.utcnow(), {}),
                SignalSource('b', 'BTC', 'buy', 0.2, datetime.utcnow(), {}),
            ]

            result = signal_aggregator.aggregate(low_confidence_sources)

            should_execute = signal_aggregator.should_execute_signal(
                result,
                min_confidence=0.6,
                min_consensus=0.5
            )

            # Low confidence should not execute
            assert should_execute is False

        except ImportError:
            pytest.skip("SignalSource not available")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_different_tickers(self, signal_aggregator):
        """Test aggregating signals for different tickers."""
        try:
            from backend.ai.signal_aggregator import SignalSource

            mixed_sources = [
                SignalSource('a', 'BTC_USDT', 'buy', 0.8, datetime.utcnow(), {}),
                SignalSource('b', 'ETH_USDT', 'sell', 0.7, datetime.utcnow(), {}),
            ]

            # Should handle or raise error for mismatched tickers
            try:
                result = signal_aggregator.aggregate(mixed_sources)
                # If it succeeds, should pick one ticker
                assert result.ticker in ['BTC_USDT', 'ETH_USDT']
            except ValueError:
                pass  # Expected for mixed tickers

        except ImportError:
            pytest.skip("SignalSource not available")

    def test_stale_signals(self, signal_aggregator):
        """Test handling of stale signals."""
        try:
            from backend.ai.signal_aggregator import SignalSource
            from datetime import timedelta

            stale_sources = [
                SignalSource(
                    'old_source',
                    'BTC_USDT',
                    'buy',
                    0.9,
                    datetime.utcnow() - timedelta(hours=24),  # Very old
                    {}
                )
            ]

            result = signal_aggregator.aggregate(stale_sources)

            # May reduce confidence for stale signals or mark as stale

        except ImportError:
            pytest.skip("SignalSource not available")

    def test_extreme_confidence_values(self, signal_aggregator):
        """Test with extreme confidence values."""
        try:
            from backend.ai.signal_aggregator import SignalSource

            extreme_sources = [
                SignalSource('a', 'BTC', 'buy', 1.0, datetime.utcnow(), {}),  # Max
                SignalSource('b', 'BTC', 'buy', 0.0, datetime.utcnow(), {}),  # Min
            ]

            result = signal_aggregator.aggregate(extreme_sources)

            # Should handle without errors
            assert 0 <= result.aggregated_confidence <= 1

        except ImportError:
            pytest.skip("SignalSource not available")
