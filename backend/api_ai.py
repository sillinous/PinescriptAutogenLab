# backend/api_ai.py
"""
AI Trading API Endpoints

New endpoints for AI/ML functionality:
- TradingView webhook processing
- Signal aggregation
- AI predictions
- Model training
- Feature engineering
"""

import asyncio
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.optimization.integrated_optimizer import start_optimization, get_optimization_status, get_strategy_config
from backend.optimization.backtester import simple_rsi_strategy, simple_ema_crossover_strategy
from backend.ab_testing.ab_service import ABTestingService, ABTestConfig, ABTestResult, get_ab_service

# AI Database functions
from backend.ai_database import (
    init_ai_schema,
    save_tradingview_signal,
    get_similar_signals,
    save_ml_model,
    log_ai_prediction
)

# TradingView integration
from backend.integrations.tradingview.webhook_handler import get_webhook_handler
from backend.integrations.tradingview.chart_service import get_chart_service

# AI/ML services - lazy imports to handle dependency issues
_signal_aggregator = None
_feature_engineer = None
_import_error = None

try:
    from backend.ai.signal_aggregator import get_signal_aggregator
    from backend.ai.features.feature_engineer import get_feature_engineer
except Exception as e:
    _import_error = str(e)
    print(f"[WARNING] AI/ML services import failed: {e}")
    print("[INFO] AI endpoints will return HTTP 503. Fix numpy/sklearn compatibility.")

    # Create dummy functions that raise HTTP errors
    def get_signal_aggregator():
        raise HTTPException(
            status_code=503,
            detail=f"AI services unavailable due to dependency error: {_import_error}"
        )

    def get_feature_engineer():
        raise HTTPException(
            status_code=503,
            detail=f"AI services unavailable due to dependency error: {_import_error}"
        )

router = APIRouter()

class OptimizationRequest(BaseModel):
    strategy_name: str
    strategy_type: str
    n_trials: int = 100

class ABTestCreateRequest(BaseModel):
    test_name: str
    variant_a_params: Dict[str, Any]
    variant_b_params: Dict[str, Any]
    min_sample_size: int = 30
    significance_level: float = 0.05

# Placeholder for historical data loading
# In a real application, this would load data from a database or external API
def load_historical_data_placeholder() -> pd.DataFrame:
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D') # Reduced to 10 days
    np.random.seed(42) # for reproducibility

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 2) + 1,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 2) - 1,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    return data





# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TradingViewWebhookRequest(BaseModel):
    """TradingView webhook payload"""
    ticker: str
    action: str
    price: Optional[float] = None
    quantity: Optional[float] = None
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    indicators: Optional[Dict[str, Any]] = None


class SignalResponse(BaseModel):
    """Response for signal processing"""
    signal_id: int
    ticker: str
    action: str
    confidence: float
    should_execute: bool
    recommended_position_size: Optional[float] = None
    reasoning: Dict[str, Any]


class ChartDataRequest(BaseModel):
    """Request for chart data"""
    ticker: str
    timeframe: str = "1h"
    bars: int = 500


class FeatureRequest(BaseModel):
    """Request for feature generation"""
    ticker: str
    timeframe: str = "1h"
    bars: int = 500


class TrainModelRequest(BaseModel):
    """Request to train RL model"""
    model_name: str
    ticker: str
    timeframe: str = "1h"
    bars: int = 1000
    total_timesteps: int = 50000


# ============================================================================
# TRADINGVIEW WEBHOOK ENDPOINTS
# ============================================================================

@router.post("/tradingview/webhook", response_model=SignalResponse)
async def process_tradingview_webhook(
    payload: TradingViewWebhookRequest,
    background_tasks: BackgroundTasks
):
    """
    Process TradingView webhook alert.

    This endpoint:
    1. Parses the TradingView alert
    2. Enriches it with market data
    3. Aggregates with other signals
    4. Determines if trade should be executed

    Example TradingView Alert Message:
    ```json
    {
        "ticker": "{{ticker}}",
        "action": "buy",
        "price": {{close}},
        "strategy": "RSI Oversold",
        "timeframe": "15m",
        "indicators": {
            "rsi": {{rsi}},
            "volume": {{volume}}
        }
    }
    ```
    """
    try:
        # Initialize AI schema if needed
        init_ai_schema()

        webhook_handler = get_webhook_handler()

        # Parse signal
        signal = webhook_handler.parse_alert(payload.dict())

        # Enrich signal with market data and AI analysis
        enriched = await webhook_handler.enrich_signal(signal)

        # Save to database
        signal_id = save_tradingview_signal(
            raw_payload=payload.dict(),
            ticker=signal.ticker,
            action=signal.action,
            price=signal.price,
            quantity=signal.quantity,
            strategy_name=signal.strategy_name,
            timeframe=signal.timeframe,
            indicators=signal.indicators,
            ai_confidence=enriched.ai_confidence,
            current_market_price=enriched.current_price
        )

        # Get signal aggregator
        aggregator = get_signal_aggregator()

        # Create signal source from TradingView
        tv_source = await aggregator.create_signal_from_tradingview(enriched)

        # For now, just use TradingView signal
        # In production, you'd aggregate with AI predictions, patterns, sentiment, etc.
        sources = [tv_source]

        # Aggregate signals
        aggregated = aggregator.aggregate(sources)

        # Determine if should execute
        should_execute = aggregator.should_execute_signal(
            aggregated,
            min_confidence=0.6,
            min_consensus=0.5
        )

        # Background task: Get similar signals for learning
        background_tasks.add_task(
            _analyze_similar_signals,
            signal.ticker,
            signal.action,
            signal.strategy_name
        )

        return SignalResponse(
            signal_id=signal_id,
            ticker=aggregated.ticker,
            action=aggregated.action,
            confidence=aggregated.aggregated_confidence,
            should_execute=should_execute,
            recommended_position_size=aggregated.recommended_position_size,
            reasoning={
                'sources': aggregated.participating_sources,
                'consensus_score': aggregated.consensus_score,
                'ai_confidence': enriched.ai_confidence,
                'market_price': enriched.current_price,
                'support_levels': enriched.support_levels,
                'resistance_levels': enriched.resistance_levels
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


async def _analyze_similar_signals(ticker: str, action: str, strategy_name: Optional[str]):
    """Background task to analyze similar historical signals"""
    try:
        similar = get_similar_signals(ticker, action, strategy_name, limit=50)
        if similar:
            avg_pnl = sum(s.get('pnl', 0) for s in similar if s.get('pnl')) / len(similar)
            win_rate = sum(1 for s in similar if s.get('success')) / len(similar)
            print(f"[INFO] Similar signals - Count: {len(similar)}, Avg P&L: {avg_pnl:.2f}, Win Rate: {win_rate:.2%}")
    except Exception as e:
        print(f"[WARNING] Failed to analyze similar signals: {e}")


# ============================================================================
# CHART DATA ENDPOINTS
# ============================================================================

@router.post("/chart/ohlcv")
async def get_chart_data(request: ChartDataRequest):
    """
    Get OHLCV chart data with technical indicators.

    Returns historical price data with calculated indicators like RSI, MACD, etc.
    """
    try:
        chart_service = get_chart_service()

        # Get OHLCV data
        df = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.bars
        )

        # Add indicators
        df_with_indicators = await chart_service.get_indicators(df)

        # Convert to dict for JSON response
        data = df_with_indicators.to_dict(orient='records')

        return {
            'ticker': request.ticker,
            'timeframe': request.timeframe,
            'bars': len(data),
            'data': data[-100:]  # Return last 100 bars to keep response size reasonable
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")


@router.get("/chart/support-resistance/{ticker}")
async def get_support_resistance(ticker: str, timeframe: str = "1h", bars: int = 200):
    """
    Calculate support and resistance levels.
    """
    try:
        chart_service = get_chart_service()

        df = await chart_service.get_ohlcv(ticker, timeframe, bars)
        levels = chart_service.calculate_support_resistance(df, num_levels=3)

        return {
            'ticker': ticker,
            'support_levels': levels['support'],
            'resistance_levels': levels['resistance'],
            'current_price': float(df.iloc[-1]['close'])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating S/R: {str(e)}")


# ============================================================================
# FEATURE ENGINEERING ENDPOINTS
# ============================================================================

@router.post("/features/generate")
async def generate_features(request: FeatureRequest):
    """
    Generate ML features from OHLCV data.

    Returns 100+ technical, statistical, and pattern-based features.
    """
    try:
        chart_service = get_chart_service()
        feature_engineer = get_feature_engineer()

        # Get chart data
        df = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.bars
        )

        # Generate features
        df_features = feature_engineer.generate_all_features(df)

        # Get feature list
        feature_columns = [col for col in df_features.columns
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return {
            'ticker': request.ticker,
            'num_features': len(feature_columns),
            'features': feature_columns,
            'sample_data': df_features[feature_columns].tail(5).to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating features: {str(e)}")


# ============================================================================
# AI MODEL ENDPOINTS
# ============================================================================

@router.post("/model/train")
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Train a new RL trading model.

    This is a long-running operation that trains a PPO agent on historical data.
    Training happens in the background.
    """
    try:
        # Add training task to background
        background_tasks.add_task(
            _train_model_background,
            request.model_name,
            request.ticker,
            request.timeframe,
            request.bars,
            request.total_timesteps
        )

        return {
            'status': 'training_started',
            'model_name': request.model_name,
            'ticker': request.ticker,
            'total_timesteps': request.total_timesteps,
            'message': 'Training started in background. Check status endpoint for progress.'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")


async def _train_model_background(
    model_name: str,
    ticker: str,
    timeframe: str,
    bars: int,
    total_timesteps: int
):
    """Background task to train model"""
    try:
        print(f"[INFO] Starting training: {model_name}")

        # Get data
        chart_service = get_chart_service()
        df = await chart_service.get_ohlcv(ticker, timeframe, bars)

        # Generate features
        feature_engineer = get_feature_engineer()
        df_features = feature_engineer.generate_all_features(df)

        # Fill NaN values
        df_features = df_features.ffill().fillna(0)

        # Train agent
        from backend.ai.reinforcement_learning.agent import train_agent

        agent = train_agent(
            df=df_features,
            model_name=model_name,
            total_timesteps=total_timesteps
        )

        # Save model metadata to database
        performance = agent.evaluate(
            agent.env,
            n_episodes=10
        )

        save_ml_model(
            name=model_name,
            model_type='RL_PPO',
            version='v1',
            file_path=str(agent.model_dir / model_name),
            hyperparameters=agent.hyperparameters,
            performance_metrics=performance
        )

        print(f"[INFO] Training completed: {model_name}")
        print(f"[INFO] Performance: {performance}")

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")


@router.get("/model/predict/{ticker}")
async def get_ai_prediction(
    ticker: str,
    model_name: str = "trading_agent_v1",
    timeframe: str = "1h"
):
    """
    Get AI prediction for a ticker.

    Returns BUY/SELL/HOLD recommendation with confidence score.
    """
    try:
        # Load model
        from backend.ai.reinforcement_learning.agent import load_trained_agent

        agent = load_trained_agent(f"models/rl/{model_name}")

        # Get current market state
        chart_service = get_chart_service()
        df = await chart_service.get_ohlcv(ticker, timeframe, bars=100)

        # Generate features
        feature_engineer = get_feature_engineer()
        df_features = feature_engineer.generate_all_features(df)
        df_features = df_features.ffill().fillna(0)

        # Prepare observation
        from backend.ai.reinforcement_learning.trading_env import create_env_from_dataframe

        env = create_env_from_dataframe(df_features)
        obs, _ = env.reset()

        # Get prediction
        prediction = agent.predict_with_reasoning(
            obs,
            market_context={
                'ticker': ticker,
                'current_price': float(df.iloc[-1]['close']),
                'timeframe': timeframe
            }
        )

        # Log prediction
        log_ai_prediction(
            model_id=1,  # TODO: Get actual model_id from database
            ticker=ticker,
            prediction_type='action',
            predicted_value=prediction,
            confidence=prediction['confidence'],
            features={},
            market_state={'price': float(df.iloc[-1]['close'])}
        )

        return prediction

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Train a model first using /api/v1/ai/model/train"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting prediction: {str(e)}")


@router.get("/signal/aggregate/{ticker}")
async def aggregate_signals(ticker: str):
    """
    Get aggregated signal from all sources.

    Combines:
    - TradingView signals
    - AI predictions
    - Technical indicators
    - Pattern recognition (future)
    - Sentiment analysis (future)
    """
    try:
        aggregator = get_signal_aggregator()
        sources = []

        # Get AI prediction
        try:
            chart_service = get_chart_service()
            df = await chart_service.get_ohlcv(ticker, "1h", 100)

            # Simple technical signal (placeholder for full AI)
            rsi = df.iloc[-1].get('rsi_14', 50)

            # Only create signal if RSI shows clear buy/sell indication
            if rsi < 30:
                action = 'buy'
                confidence = min((30 - rsi) / 30, 1.0)
                ai_signal = await aggregator.create_signal_from_ai(
                    ticker=ticker,
                    action=action,
                    confidence=confidence,
                    model_name='technical_indicators',
                    reasoning={'rsi': float(rsi), 'signal_type': 'oversold'}
                )
                sources.append(ai_signal)
            elif rsi > 70:
                action = 'sell'
                confidence = min((rsi - 70) / 30, 1.0)
                ai_signal = await aggregator.create_signal_from_ai(
                    ticker=ticker,
                    action=action,
                    confidence=confidence,
                    model_name='technical_indicators',
                    reasoning={'rsi': float(rsi), 'signal_type': 'overbought'}
                )
                sources.append(ai_signal)
            # else: neutral zone (30-70), no signal created

        except Exception as e:
            print(f"[WARNING] Could not get AI signal: {e}")

        # If no signals (RSI in neutral zone), return neutral response
        if not sources:
            from datetime import datetime
            return {
                'ticker': ticker,
                'action': 'NEUTRAL',
                'confidence': 0.0,
                'consensus': 0.0,
                'sources': [],
                'recommended_position_size': None,
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'No strong signals detected - market in neutral zone'
            }

        # Aggregate
        aggregated = aggregator.aggregate(sources)

        return {
            'ticker': aggregated.ticker,
            'action': aggregated.action,
            'confidence': aggregated.aggregated_confidence,
            'consensus': aggregated.consensus_score,
            'sources': aggregated.participating_sources,
            'recommended_position_size': aggregated.recommended_position_size,
            'timestamp': aggregated.timestamp.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aggregating signals: {str(e)}")

@router.post("/optimize/start")
async def start_strategy_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Starts an asynchronous strategy optimization process using Optuna.
    """
    if get_strategy_config(request.strategy_type) is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy type: {request.strategy_type}")

    # Load historical data (using placeholder for now)
    historical_data = load_historical_data_placeholder()

    # Run optimization in a background task
    background_tasks.add_task(
        start_optimization,
        strategy_name=request.strategy_name,
        strategy_type=request.strategy_type,
        data=historical_data,
        n_trials=request.n_trials
    )

    return {"message": f"Optimization for strategy '{request.strategy_name}' started successfully.",
            "strategy_name": request.strategy_name,
            "status_endpoint": f"/api/v1/ai/optimize/status/{request.strategy_name}"}

@router.get("/optimize/status/{strategy_name}")
async def get_strategy_optimization_status(strategy_name: str):
    """
    Retrieves the current status of an ongoing or completed strategy optimization.
    """
    status = get_optimization_status(strategy_name)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Optimization for strategy '{strategy_name}' not found or not started.")
    return status

@router.get("/optimize/strategies")
async def get_available_strategies():
    """
    Returns a list of pre-configured strategies available for optimization.
    """
    from backend.optimization.integrated_optimizer import STRATEGY_CONFIGS
    return [{"name": config["name"], "type": strategy_type} for strategy_type, config in STRATEGY_CONFIGS.items()]

@router.post("/abtest/create")
async def create_ab_test_endpoint(request: ABTestCreateRequest):
    """
    Creates a new A/B test for comparing two strategy variants.
    """
    ab_service = get_ab_service()
    try:
        test_config = ab_service.create_test(
            test_name=request.test_name,
            variant_a_params=request.variant_a_params,
            variant_b_params=request.variant_b_params,
            min_sample_size=request.min_sample_size,
            significance_level=request.significance_level
        )
        return {"message": f"A/B test '{request.test_name}' created successfully.", "test_config": test_config}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/abtest/results/{test_name}")
async def get_ab_test_results_endpoint(test_name: str):
    """
    Retrieves the current results and statistical analysis for a given A/B test.
    """
    ab_service = get_ab_service()
    try:
        results = ab_service.get_test_results(test_name)
        return results
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/abtest/promote/{test_name}")
async def promote_ab_test_winner_endpoint(test_name: str):
    """
    Promotes the winning variant of a completed A/B test to be the new best strategy.
    """
    ab_service = get_ab_service()
    try:
        promotion_details = ab_service.promote_winner(test_name)
        return {"message": f"Winner of A/B test '{test_name}' promoted successfully.", "details": promotion_details}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to promote winner: {str(e)}")

@router.get("/abtest/active")
async def get_active_ab_tests_endpoint():
    """
    Retrieves a list of all currently active A/B tests.
    """
    ab_service = get_ab_service()
    active_tests = ab_service.get_active_tests()
    return {"active_tests": active_tests}




# ============================================================================
# INITIALIZE
# ============================================================================

@router.on_event("startup")
async def startup_event():
    """Initialize AI components on startup"""
    try:
        init_ai_schema()
        print("[INFO] AI database schema initialized")
    except Exception as e:
        print(f"[WARNING] Failed to initialize AI schema: {e}")
