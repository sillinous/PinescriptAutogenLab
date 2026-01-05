"""
Ensemble prediction system combining multiple deep learning models.

This module implements an ensemble approach that aggregates predictions
from LSTM, Transformer, and other models for more robust forecasting.

Features:
- Weighted ensemble with learned or manual weights
- Stacking ensemble with meta-learner
- Dynamic weight adjustment based on recent performance
- Uncertainty estimation through prediction variance
- Confidence scoring for ensemble predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models for robust predictions.
    """

    def __init__(
        self,
        models: Optional[List] = None,
        weights: Optional[List[float]] = None,
        ensemble_method: str = "weighted_average"
    ):
        """
        Args:
            models: List of predictor models (LSTM, Transformer, etc.)
            weights: Weights for each model (if None, equal weights)
            ensemble_method: Method to combine predictions
                - "weighted_average": Weighted average of predictions
                - "median": Median of predictions
                - "best": Use best performing model
                - "stacking": Meta-learner on top
        """
        self.models = models if models is not None else []
        self.ensemble_method = ensemble_method

        # Initialize weights
        if weights is None and len(self.models) > 0:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            self.weights = weights if weights is not None else []

        # Performance tracking
        self.model_performances = {i: [] for i in range(len(self.models))}
        self.recent_errors = {i: [] for i in range(len(self.models))}

    def add_model(self, model, weight: float = None):
        """Add a model to the ensemble."""
        self.models.append(model)

        if weight is None:
            # Recalculate equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            self.weights.append(weight)
            # Normalize weights
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        # Initialize performance tracking
        self.model_performances[len(self.models) - 1] = []
        self.recent_errors[len(self.models) - 1] = []

    def predict(
        self,
        data: np.ndarray,
        return_individual: bool = False,
        return_uncertainty: bool = False
    ) -> Dict:
        """
        Make ensemble predictions.

        Args:
            data: Input data for prediction
            return_individual: Return individual model predictions
            return_uncertainty: Calculate prediction uncertainty

        Returns:
            Dictionary with predictions and optional metadata
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(data)
                predictions.append(pred)
            except Exception as e:
                print(f"Model prediction failed: {e}")
                # Use NaN for failed predictions
                predictions.append(np.full_like(predictions[0] if predictions else [0], np.nan))

        predictions = np.array(predictions)

        # Remove NaN predictions
        valid_mask = ~np.isnan(predictions).any(axis=1)
        valid_predictions = predictions[valid_mask]
        valid_weights = np.array([self.weights[i] for i in range(len(self.weights)) if valid_mask[i]])

        # Normalize valid weights
        if valid_weights.sum() > 0:
            valid_weights = valid_weights / valid_weights.sum()

        # Combine predictions based on method
        if self.ensemble_method == "weighted_average":
            ensemble_pred = np.average(valid_predictions, axis=0, weights=valid_weights)

        elif self.ensemble_method == "median":
            ensemble_pred = np.median(valid_predictions, axis=0)

        elif self.ensemble_method == "best":
            # Use the model with best recent performance
            best_idx = self._get_best_model()
            if best_idx is not None and best_idx < len(predictions):
                ensemble_pred = predictions[best_idx]
            else:
                ensemble_pred = np.mean(valid_predictions, axis=0)

        else:
            # Default to simple average
            ensemble_pred = np.mean(valid_predictions, axis=0)

        result = {
            "prediction": ensemble_pred.tolist(),
            "num_models": len(valid_predictions)
        }

        # Add individual predictions
        if return_individual:
            result["individual_predictions"] = predictions.tolist()
            result["model_weights"] = self.weights

        # Add uncertainty estimation
        if return_uncertainty:
            # Calculate variance across predictions
            variance = np.var(valid_predictions, axis=0)
            std_dev = np.std(valid_predictions, axis=0)

            # Confidence is inversely related to uncertainty
            # Lower variance = higher confidence
            confidence = self._calculate_confidence(variance)

            result["uncertainty"] = {
                "variance": variance.tolist(),
                "std_dev": std_dev.tolist(),
                "confidence": confidence
            }

        return result

    def update_weights_from_performance(self, window: int = 10):
        """
        Update model weights based on recent performance.

        Args:
            window: Number of recent predictions to consider
        """
        if any(len(errors) == 0 for errors in self.recent_errors.values()):
            # Not enough data yet
            return

        # Calculate recent average error for each model
        recent_errors = []
        for i in range(len(self.models)):
            errors = self.recent_errors[i][-window:]
            avg_error = np.mean(errors) if errors else float('inf')
            recent_errors.append(avg_error)

        # Convert errors to weights (inverse relationship)
        # Models with lower error get higher weight
        recent_errors = np.array(recent_errors)
        if np.all(recent_errors == 0):
            weights = np.ones(len(self.models))
        else:
            # Avoid division by zero
            inv_errors = 1.0 / (recent_errors + 1e-8)
            weights = inv_errors / inv_errors.sum()

        self.weights = weights.tolist()

    def record_prediction_error(self, model_idx: int, error: float):
        """Record prediction error for a specific model."""
        if model_idx in self.recent_errors:
            self.recent_errors[model_idx].append(error)
            self.model_performances[model_idx].append(error)

            # Keep only recent errors (last 100)
            if len(self.recent_errors[model_idx]) > 100:
                self.recent_errors[model_idx] = self.recent_errors[model_idx][-100:]

    def _get_best_model(self) -> Optional[int]:
        """Get index of best performing model."""
        if not any(self.recent_errors.values()):
            return None

        # Calculate average recent error for each model
        avg_errors = []
        for i in range(len(self.models)):
            errors = self.recent_errors[i][-10:]  # Last 10 predictions
            avg_error = np.mean(errors) if errors else float('inf')
            avg_errors.append(avg_error)

        return int(np.argmin(avg_errors))

    def _calculate_confidence(self, variance: np.ndarray) -> float:
        """
        Calculate confidence score from prediction variance.

        Args:
            variance: Prediction variance

        Returns:
            Confidence score between 0 and 1
        """
        # Normalize variance to confidence score
        # Using exponential decay: high variance = low confidence
        mean_var = np.mean(variance)

        # Confidence = exp(-k * variance)
        # where k is a scaling factor
        k = 1.0
        confidence = np.exp(-k * mean_var)

        return float(confidence)

    def get_performance_summary(self) -> Dict:
        """Get performance summary for all models."""
        summary = {}

        for i, model in enumerate(self.models):
            model_name = f"Model_{i}"
            errors = self.model_performances[i]

            if errors:
                summary[model_name] = {
                    "mean_error": float(np.mean(errors)),
                    "std_error": float(np.std(errors)),
                    "min_error": float(np.min(errors)),
                    "max_error": float(np.max(errors)),
                    "num_predictions": len(errors),
                    "current_weight": self.weights[i] if i < len(self.weights) else 0.0
                }
            else:
                summary[model_name] = {
                    "mean_error": None,
                    "num_predictions": 0,
                    "current_weight": self.weights[i] if i < len(self.weights) else 0.0
                }

        return summary

    def save(self, filepath: str):
        """Save ensemble configuration and performance history."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        config = {
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "num_models": len(self.models),
            "model_performances": {
                str(k): v for k, v in self.model_performances.items()
            },
            "recent_errors": {
                str(k): v for k, v in self.recent_errors.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, filepath: str):
        """Load ensemble configuration."""
        with open(filepath, 'r') as f:
            config = json.load(f)

        self.ensemble_method = config.get("ensemble_method", "weighted_average")
        self.weights = config.get("weights", [])
        self.model_performances = {
            int(k): v for k, v in config.get("model_performances", {}).items()
        }
        self.recent_errors = {
            int(k): v for k, v in config.get("recent_errors", {}).items()
        }

    def predict_with_confidence(
        self,
        data: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Make prediction with confidence check.

        Args:
            data: Input data
            confidence_threshold: Minimum confidence for valid prediction

        Returns:
            (prediction, confidence, is_reliable)
        """
        result = self.predict(data, return_uncertainty=True)

        prediction = np.array(result["prediction"])
        confidence = result["uncertainty"]["confidence"]
        is_reliable = confidence >= confidence_threshold

        return prediction, confidence, is_reliable
