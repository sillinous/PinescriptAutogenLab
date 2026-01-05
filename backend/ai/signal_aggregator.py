# backend/ai/signal_aggregator.py
"""
Signal Aggregation Service

Combines signals from multiple sources and uses AI to weight them:
- TradingView user alerts
- AI-generated signals (from RL agent, pattern recognition)
- Technical indicator signals
- Sentiment-based signals

The aggregator learns optimal weights over time based on signal performance.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import numpy as np
from backend.integrations.tradingview.webhook_handler import TradingSignal, EnrichedSignal


class SignalSource(BaseModel):
    """Represents a signal from a specific source"""
    source_type: str  # 'tradingview', 'ai_agent', 'pattern_recognition', 'sentiment'
    source_name: str  # Specific identifier
    signal: TradingSignal
    confidence: float  # 0.0 to 1.0
    weight: float = 1.0  # Learned weight for this source
    metadata: Dict[str, Any] = {}


class AggregatedSignal(BaseModel):
    """Combined signal from multiple sources"""
    ticker: str
    action: str  # 'buy', 'sell', 'hold'
    aggregated_confidence: float  # Final confidence score
    participating_sources: List[str]  # List of sources that contributed
    source_signals: List[SignalSource]  # All individual signals
    consensus_score: float  # How much sources agree (0-1)
    recommended_position_size: Optional[float] = None  # Based on confidence
    timestamp: datetime = None

    def __init__(self, **data):
        if data.get('timestamp') is None:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class SignalAggregator:
    """
    Combines and weighs signals from multiple sources using learned weights.

    The aggregator maintains weights for each signal source based on
    historical performance. Better-performing sources get higher weights.
    """

    def __init__(self):
        """Initialize with equal weights, which will be learned over time."""
        # Source weights (learned from performance)
        self.source_weights: Dict[str, float] = {
            'tradingview': 1.0,
            'ai_agent': 1.2,  # Slightly higher initial weight for AI
            'pattern_recognition': 1.0,
            'sentiment': 0.8,
            'technical_indicators': 0.9
        }

        # Historical performance tracking
        self.source_performance: Dict[str, Dict[str, Any]] = {}

    def aggregate(self, signals: List[SignalSource]) -> AggregatedSignal:
        """
        Combine multiple signals into a single aggregated signal.

        Args:
            signals: List of signals from different sources

        Returns:
            AggregatedSignal with weighted combination
        """
        if not signals:
            raise ValueError("Cannot aggregate empty signal list")

        # Group by ticker and action
        ticker = signals[0].signal.ticker
        action_votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        total_weight = 0.0

        for source_signal in signals:
            action = source_signal.signal.action
            if action in ['long']:
                action = 'buy'
            elif action in ['short', 'exit', 'close']:
                action = 'sell'

            # Get source weight
            weight = self.source_weights.get(source_signal.source_type, 1.0)

            # Weighted vote
            vote_strength = source_signal.confidence * weight
            action_votes[action] += vote_strength
            total_weight += weight

        # Determine winning action
        winning_action = max(action_votes, key=action_votes.get)
        winning_votes = action_votes[winning_action]

        # Calculate aggregated confidence
        if total_weight > 0:
            aggregated_confidence = winning_votes / total_weight
        else:
            aggregated_confidence = 0.5

        # Calculate consensus (how much sources agree)
        total_votes = sum(action_votes.values())
        if total_votes > 0:
            consensus_score = winning_votes / total_votes
        else:
            consensus_score = 0.0

        # Extract participating sources
        participating = [s.source_name for s in signals]

        # Calculate recommended position size based on confidence
        position_size = self._calculate_position_size(aggregated_confidence, consensus_score)

        return AggregatedSignal(
            ticker=ticker,
            action=winning_action,
            aggregated_confidence=aggregated_confidence,
            participating_sources=participating,
            source_signals=signals,
            consensus_score=consensus_score,
            recommended_position_size=position_size
        )

    def _calculate_position_size(self, confidence: float, consensus: float) -> float:
        """
        Calculate recommended position size based on confidence and consensus.

        Uses Kelly Criterion-inspired approach:
        - High confidence + high consensus = larger position
        - Low confidence or low consensus = smaller position

        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        # Base position size from confidence
        base_size = confidence

        # Adjust for consensus
        consensus_factor = 0.5 + (consensus * 0.5)  # 0.5 to 1.0

        # Combined position size
        position_size = base_size * consensus_factor

        # Cap at reasonable limits
        position_size = max(0.1, min(1.0, position_size))

        return position_size

    def update_source_weight(
        self,
        source_type: str,
        source_name: str,
        performance_score: float
    ):
        """
        Update weight for a signal source based on performance.

        Args:
            source_type: Type of source ('tradingview', 'ai_agent', etc.)
            source_name: Specific source identifier
            performance_score: Performance metric (e.g., win rate, Sharpe ratio)
        """
        # Track performance
        key = f"{source_type}:{source_name}"
        if key not in self.source_performance:
            self.source_performance[key] = {
                'scores': [],
                'avg_score': 0.5
            }

        self.source_performance[key]['scores'].append(performance_score)

        # Keep last 100 scores
        if len(self.source_performance[key]['scores']) > 100:
            self.source_performance[key]['scores'] = \
                self.source_performance[key]['scores'][-100:]

        # Calculate average
        avg_score = np.mean(self.source_performance[key]['scores'])
        self.source_performance[key]['avg_score'] = avg_score

        # Update weight for source type
        # Weight = 0.5 + (avg_score * 1.5)
        # So 0% performance = 0.5 weight, 100% performance = 2.0 weight
        self.source_weights[source_type] = 0.5 + (avg_score * 1.5)

    def get_source_statistics(self, source_type: str) -> Dict[str, Any]:
        """Get performance statistics for a source type."""
        matching_sources = [
            (key, stats) for key, stats in self.source_performance.items()
            if key.startswith(f"{source_type}:")
        ]

        if not matching_sources:
            return {
                'weight': self.source_weights.get(source_type, 1.0),
                'avg_performance': 0.5,
                'num_signals': 0
            }

        all_scores = []
        for _, stats in matching_sources:
            all_scores.extend(stats['scores'])

        return {
            'weight': self.source_weights.get(source_type, 1.0),
            'avg_performance': np.mean(all_scores) if all_scores else 0.5,
            'num_signals': len(all_scores),
            'min_performance': np.min(all_scores) if all_scores else 0.0,
            'max_performance': np.max(all_scores) if all_scores else 1.0
        }

    def should_execute_signal(
        self,
        aggregated_signal: AggregatedSignal,
        min_confidence: float = 0.6,
        min_consensus: float = 0.5
    ) -> bool:
        """
        Determine if aggregated signal should be executed.

        Args:
            aggregated_signal: The aggregated signal
            min_confidence: Minimum confidence threshold
            min_consensus: Minimum consensus threshold

        Returns:
            True if signal should be executed, False otherwise
        """
        return (
            aggregated_signal.aggregated_confidence >= min_confidence and
            aggregated_signal.consensus_score >= min_consensus and
            aggregated_signal.action != 'hold'
        )

    async def create_signal_from_tradingview(
        self,
        tv_signal: EnrichedSignal
    ) -> SignalSource:
        """Convert TradingView signal to SignalSource."""
        confidence = tv_signal.ai_confidence or 0.7

        return SignalSource(
            source_type='tradingview',
            source_name=tv_signal.signal.strategy_name or 'default',
            signal=tv_signal.signal,
            confidence=confidence,
            metadata={
                'support_levels': tv_signal.support_levels,
                'resistance_levels': tv_signal.resistance_levels,
                'trend': tv_signal.trend_direction
            }
        )

    async def create_signal_from_ai(
        self,
        ticker: str,
        action: str,
        confidence: float,
        model_name: str,
        reasoning: Dict[str, Any]
    ) -> SignalSource:
        """Create signal from AI model prediction."""
        signal = TradingSignal(
            ticker=ticker,
            action=action,
            strategy_name=f"AI:{model_name}"
        )

        return SignalSource(
            source_type='ai_agent',
            source_name=model_name,
            signal=signal,
            confidence=confidence,
            metadata={
                'reasoning': reasoning,
                'model_version': reasoning.get('version', 'v1')
            }
        )

    async def create_signal_from_pattern(
        self,
        ticker: str,
        pattern_type: str,
        confidence: float,
        action: str
    ) -> SignalSource:
        """Create signal from pattern recognition."""
        signal = TradingSignal(
            ticker=ticker,
            action=action,
            strategy_name=f"Pattern:{pattern_type}"
        )

        return SignalSource(
            source_type='pattern_recognition',
            source_name=pattern_type,
            signal=signal,
            confidence=confidence,
            metadata={'pattern': pattern_type}
        )

    async def create_signal_from_sentiment(
        self,
        ticker: str,
        sentiment_score: float,
        sources: List[str]
    ) -> SignalSource:
        """
        Create signal from sentiment analysis.

        Args:
            ticker: Symbol
            sentiment_score: -1 (very bearish) to +1 (very bullish)
            sources: List of sentiment sources (Twitter, Reddit, news, etc.)
        """
        # Convert sentiment to action
        if sentiment_score > 0.3:
            action = 'buy'
        elif sentiment_score < -0.3:
            action = 'sell'
        else:
            action = 'hold'

        # Convert sentiment to confidence
        confidence = abs(sentiment_score)

        signal = TradingSignal(
            ticker=ticker,
            action=action,
            strategy_name=f"Sentiment:{','.join(sources)}"
        )

        return SignalSource(
            source_type='sentiment',
            source_name='aggregated',
            signal=signal,
            confidence=confidence,
            metadata={
                'sentiment_score': sentiment_score,
                'sources': sources
            }
        )


# Singleton instance
_signal_aggregator: Optional[SignalAggregator] = None


def get_signal_aggregator() -> SignalAggregator:
    """Get or create signal aggregator singleton."""
    global _signal_aggregator
    if _signal_aggregator is None:
        _signal_aggregator = SignalAggregator()
    return _signal_aggregator
