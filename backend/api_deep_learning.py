"""
FastAPI routes for Phase 2: Deep Learning models.

This module provides API endpoints for:
- LSTM price prediction training and inference
- Transformer model training and inference
- Ensemble predictions combining multiple models
- Model performance tracking and comparison
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import asyncio

from backend.ai.deep_learning.lstm_predictor import LSTMPredictor
from backend.ai.deep_learning.transformer_predictor import TransformerPredictor
from backend.ai.deep_learning.ensemble import EnsemblePredictor
from backend.integrations.tradingview.chart_service import ChartDataService

router = APIRouter(prefix="/api/v2/deep-learning", tags=["Deep Learning"])

# Global storage for models (in production, use Redis or database)
models_cache = {
    "lstm": {},  # {ticker: LSTMPredictor}
    "transformer": {},  # {ticker: TransformerPredictor}
    "ensemble": {}  # {ticker: EnsemblePredictor}
}

# Service instances
chart_service = ChartDataService()

# Model storage directory
MODEL_DIR = Path("models/deep_learning")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Request/Response Models
# ============================================================================

class TrainLSTMRequest(BaseModel):
    ticker: str
    timeframe: str = "1h"
    lookback_days: int = 60
    sequence_length: int = 60
    prediction_horizon: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10


class TrainTransformerRequest(BaseModel):
    ticker: str
    timeframe: str = "1h"
    lookback_days: int = 60
    sequence_length: int = 60
    prediction_horizon: int = 1
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 3
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10


class PredictRequest(BaseModel):
    ticker: str
    model_type: str  # "lstm", "transformer", "ensemble"
    timeframe: str = "1h"
    sequence_length: int = 60


class EnsembleConfigRequest(BaseModel):
    ticker: str
    model_types: List[str]  # ["lstm", "transformer"]
    weights: Optional[List[float]] = None
    ensemble_method: str = "weighted_average"


class PredictionResponse(BaseModel):
    ticker: str
    model_type: str
    prediction: List[float]
    confidence: Optional[float] = None
    uncertainty: Optional[Dict] = None
    timestamp: str
    sequence_length: int
    prediction_horizon: int


# ============================================================================
# LSTM Endpoints
# ============================================================================

@router.post("/lstm/train")
async def train_lstm_model(request: TrainLSTMRequest, background_tasks: BackgroundTasks):
    """
    Train an LSTM model for price prediction.

    This endpoint trains a new LSTM model on historical price data.
    Training runs in the background and the model is cached for inference.

    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks

    Returns:
        Training job status and metadata
    """
    try:
        # Fetch historical data
        data = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.lookback_days * 24 if request.timeframe == "1h" else request.lookback_days
        )

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")

        # Select OHLCV columns
        ohlcv_data = data[['open', 'high', 'low', 'close', 'volume']]

        # Validate sufficient data
        min_required = request.sequence_length + request.prediction_horizon + 10
        if len(ohlcv_data) < min_required:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for training. Got {len(ohlcv_data)} rows, "
                       f"but need at least {min_required} (sequence_length={request.sequence_length} + "
                       f"prediction_horizon={request.prediction_horizon} + buffer). "
                       f"Try reducing lookback_days, sequence_length, or prediction_horizon."
            )

        # Initialize LSTM predictor
        predictor = LSTMPredictor(
            input_size=5,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            sequence_length=request.sequence_length,
            prediction_horizon=request.prediction_horizon,
            learning_rate=request.learning_rate
        )

        # Prepare data
        train_loader, val_loader = predictor.prepare_data(
            ohlcv_data,
            train_split=0.8,
            batch_size=request.batch_size
        )

        # Train model
        training_result = predictor.train(
            train_loader,
            val_loader,
            epochs=request.epochs,
            early_stopping_patience=request.early_stopping_patience,
            verbose=True
        )

        # Save model
        model_path = MODEL_DIR / f"lstm_{request.ticker}_{request.timeframe}.pt"
        predictor.save(str(model_path))

        # Cache model
        models_cache["lstm"][request.ticker] = predictor

        return {
            "status": "completed",
            "ticker": request.ticker,
            "model_type": "LSTM",
            "training_result": training_result,
            "model_path": str(model_path),
            "config": {
                "sequence_length": request.sequence_length,
                "prediction_horizon": request.prediction_horizon,
                "hidden_size": request.hidden_size,
                "num_layers": request.num_layers
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM training failed: {str(e)}")


@router.post("/lstm/predict", response_model=PredictionResponse)
async def predict_with_lstm(request: PredictRequest):
    """
    Make predictions using a trained LSTM model.

    Args:
        request: Prediction configuration

    Returns:
        Price predictions with confidence scores
    """
    try:
        # Load model from cache or disk
        if request.ticker not in models_cache["lstm"]:
            model_path = MODEL_DIR / f"lstm_{request.ticker}_{request.timeframe}.pt"
            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"No trained LSTM model found for {request.ticker}. Train a model first."
                )

            predictor = LSTMPredictor()
            predictor.load(str(model_path))
            models_cache["lstm"][request.ticker] = predictor
        else:
            predictor = models_cache["lstm"][request.ticker]

        # Fetch recent data for prediction
        data = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.sequence_length + 10
        )

        if len(data) < request.sequence_length:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")

        # Prepare input data
        ohlcv_data = data[['open', 'high', 'low', 'close', 'volume']].values[-request.sequence_length:]

        # Make prediction
        prediction = predictor.predict(ohlcv_data)

        return PredictionResponse(
            ticker=request.ticker,
            model_type="LSTM",
            prediction=prediction.tolist(),
            timestamp=datetime.now().isoformat(),
            sequence_length=request.sequence_length,
            prediction_horizon=predictor.prediction_horizon
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM prediction failed: {str(e)}")


# ============================================================================
# Transformer Endpoints
# ============================================================================

@router.post("/transformer/train")
async def train_transformer_model(request: TrainTransformerRequest):
    """
    Train a Transformer model for price prediction.

    Args:
        request: Training configuration

    Returns:
        Training job status and metadata
    """
    try:
        # Fetch historical data
        data = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.lookback_days * 24 if request.timeframe == "1h" else request.lookback_days
        )

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")

        # Select OHLCV columns
        ohlcv_data = data[['open', 'high', 'low', 'close', 'volume']]

        # Initialize Transformer predictor
        predictor = TransformerPredictor(
            input_size=5,
            d_model=request.d_model,
            nhead=request.nhead,
            num_encoder_layers=request.num_encoder_layers,
            sequence_length=request.sequence_length,
            prediction_horizon=request.prediction_horizon,
            learning_rate=request.learning_rate
        )

        # Prepare data
        train_loader, val_loader = predictor.prepare_data(
            ohlcv_data,
            train_split=0.8,
            batch_size=request.batch_size
        )

        # Train model
        training_result = predictor.train(
            train_loader,
            val_loader,
            epochs=request.epochs,
            early_stopping_patience=request.early_stopping_patience,
            verbose=True
        )

        # Save model
        model_path = MODEL_DIR / f"transformer_{request.ticker}_{request.timeframe}.pt"
        predictor.save(str(model_path))

        # Cache model
        models_cache["transformer"][request.ticker] = predictor

        return {
            "status": "completed",
            "ticker": request.ticker,
            "model_type": "Transformer",
            "training_result": training_result,
            "model_path": str(model_path),
            "config": {
                "sequence_length": request.sequence_length,
                "prediction_horizon": request.prediction_horizon,
                "d_model": request.d_model,
                "nhead": request.nhead,
                "num_encoder_layers": request.num_encoder_layers
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformer training failed: {str(e)}")


@router.post("/transformer/predict", response_model=PredictionResponse)
async def predict_with_transformer(request: PredictRequest):
    """
    Make predictions using a trained Transformer model.

    Args:
        request: Prediction configuration

    Returns:
        Price predictions
    """
    try:
        # Load model from cache or disk
        if request.ticker not in models_cache["transformer"]:
            model_path = MODEL_DIR / f"transformer_{request.ticker}_{request.timeframe}.pt"
            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"No trained Transformer model found for {request.ticker}. Train a model first."
                )

            predictor = TransformerPredictor()
            predictor.load(str(model_path))
            models_cache["transformer"][request.ticker] = predictor
        else:
            predictor = models_cache["transformer"][request.ticker]

        # Fetch recent data
        data = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.sequence_length + 10
        )

        if len(data) < request.sequence_length:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")

        # Prepare input data
        ohlcv_data = data[['open', 'high', 'low', 'close', 'volume']].values[-request.sequence_length:]

        # Make prediction
        prediction = predictor.predict(ohlcv_data)

        return PredictionResponse(
            ticker=request.ticker,
            model_type="Transformer",
            prediction=prediction.tolist(),
            timestamp=datetime.now().isoformat(),
            sequence_length=request.sequence_length,
            prediction_horizon=predictor.prediction_horizon
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformer prediction failed: {str(e)}")


# ============================================================================
# Ensemble Endpoints
# ============================================================================

@router.post("/ensemble/create")
async def create_ensemble(request: EnsembleConfigRequest):
    """
    Create an ensemble of multiple models.

    Args:
        request: Ensemble configuration

    Returns:
        Ensemble creation status
    """
    try:
        models = []

        # Load requested models
        for model_type in request.model_types:
            if model_type == "lstm":
                if request.ticker in models_cache["lstm"]:
                    models.append(models_cache["lstm"][request.ticker])
                else:
                    model_path = MODEL_DIR / f"lstm_{request.ticker}_1h.pt"
                    if model_path.exists():
                        predictor = LSTMPredictor()
                        predictor.load(str(model_path))
                        models.append(predictor)

            elif model_type == "transformer":
                if request.ticker in models_cache["transformer"]:
                    models.append(models_cache["transformer"][request.ticker])
                else:
                    model_path = MODEL_DIR / f"transformer_{request.ticker}_1h.pt"
                    if model_path.exists():
                        predictor = TransformerPredictor()
                        predictor.load(str(model_path))
                        models.append(predictor)

        if len(models) == 0:
            raise HTTPException(
                status_code=404,
                detail="No trained models found. Train models first before creating ensemble."
            )

        # Create ensemble
        ensemble = EnsemblePredictor(
            models=models,
            weights=request.weights,
            ensemble_method=request.ensemble_method
        )

        # Cache ensemble
        models_cache["ensemble"][request.ticker] = ensemble

        return {
            "status": "created",
            "ticker": request.ticker,
            "num_models": len(models),
            "model_types": request.model_types,
            "ensemble_method": request.ensemble_method,
            "weights": ensemble.weights,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble creation failed: {str(e)}")


@router.post("/ensemble/predict", response_model=PredictionResponse)
async def predict_with_ensemble(request: PredictRequest):
    """
    Make predictions using ensemble of models.

    Args:
        request: Prediction configuration

    Returns:
        Ensemble predictions with uncertainty estimates
    """
    try:
        # Check if ensemble exists
        if request.ticker not in models_cache["ensemble"]:
            raise HTTPException(
                status_code=404,
                detail=f"No ensemble found for {request.ticker}. Create ensemble first."
            )

        ensemble = models_cache["ensemble"][request.ticker]

        # Fetch recent data
        data = await chart_service.get_ohlcv(
            symbol=request.ticker,
            timeframe=request.timeframe,
            bars=request.sequence_length + 10
        )

        if len(data) < request.sequence_length:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")

        # Prepare input data
        ohlcv_data = data[['open', 'high', 'low', 'close', 'volume']].values[-request.sequence_length:]

        # Make ensemble prediction
        result = ensemble.predict(
            ohlcv_data,
            return_individual=False,
            return_uncertainty=True
        )

        return PredictionResponse(
            ticker=request.ticker,
            model_type="Ensemble",
            prediction=result["prediction"],
            confidence=result.get("uncertainty", {}).get("confidence"),
            uncertainty=result.get("uncertainty"),
            timestamp=datetime.now().isoformat(),
            sequence_length=request.sequence_length,
            prediction_horizon=1
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.get("/models/list")
async def list_trained_models():
    """List all trained models."""
    models = {
        "lstm": [],
        "transformer": [],
        "ensemble": list(models_cache["ensemble"].keys())
    }

    # Scan model directory
    for model_file in MODEL_DIR.glob("*.pt"):
        parts = model_file.stem.split("_")
        if len(parts) >= 2:
            model_type = parts[0]
            ticker = parts[1]

            if model_type in models:
                models[model_type].append({
                    "ticker": ticker,
                    "file": model_file.name,
                    "size_mb": model_file.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })

    return models


@router.delete("/models/{model_type}/{ticker}")
async def delete_model(model_type: str, ticker: str):
    """Delete a trained model."""
    try:
        # Remove from cache
        if ticker in models_cache.get(model_type, {}):
            del models_cache[model_type][ticker]

        # Remove from disk
        model_path = MODEL_DIR / f"{model_type}_{ticker}_1h.pt"
        if model_path.exists():
            model_path.unlink()

        return {
            "status": "deleted",
            "model_type": model_type,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model deletion failed: {str(e)}")


@router.get("/models/performance/{ticker}")
async def get_model_performance(ticker: str):
    """Get performance comparison of all models for a ticker."""
    try:
        performance = {}

        # Check ensemble performance
        if ticker in models_cache["ensemble"]:
            ensemble = models_cache["ensemble"][ticker]
            performance["ensemble"] = ensemble.get_performance_summary()

        return {
            "ticker": ticker,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance retrieval failed: {str(e)}")
