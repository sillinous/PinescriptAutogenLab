# backend/optimization/optuna_service.py

import optuna
from optuna.trial import Trial
from typing import Dict, Any, Callable, Optional, List
import json
from datetime import datetime, timedelta
from pathlib import Path
from backend.config import Config
from backend.database import get_db
import sqlite3


class StrategyOptimizer:
    """Optuna-powered strategy parameter optimization."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.study: Optional[optuna.Study] = None
        self.best_params: Dict[str, Any] = {}

        # Storage path for Optuna studies
        self.storage_dir = Path(Config.DATA_DIR) / "optuna"
        self.storage_dir.mkdir(exist_ok=True, parents=True)

        # SQLite storage for persistence
        self.storage_url = f"sqlite:///{self.storage_dir / f'{strategy_name}.db'}"

    def create_study(
        self,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None
    ) -> optuna.Study:
        """
        Create or load Optuna study.

        Args:
            direction: 'maximize' or 'minimize'
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
        """
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)

        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )

        self.study = optuna.create_study(
            study_name=self.strategy_name,
            storage=self.storage_url,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )

        return self.study

    def optimize(
        self,
        objective_func: Callable[[Trial], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            objective_func: Function that takes Trial and returns metric to optimize
            n_trials: Number of trials to run
            timeout: Timeout in seconds
            callbacks: List of callback functions
            show_progress: Show progress bar

        Returns:
            Dictionary with best params and optimization results
        """
        if self.study is None:
            self.create_study()

        # Run optimization
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=show_progress
        )

        # Get best parameters
        self.best_params = self.study.best_params

        # Save to database
        self._save_best_params()

        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials),
            'study_name': self.strategy_name
        }

    def get_param_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using fANOVA."""
        if self.study is None or len(self.study.trials) < 10:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            print(f"[WARN] Could not calculate param importance: {e}")
            return {}

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history for visualization."""
        if self.study is None:
            return []

        history = []
        for trial in self.study.trials:
            history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
            })

        return history

    def _save_best_params(self):
        """Save best parameters to database."""
        conn = get_db()
        cursor = conn.cursor()

        for param_name, param_value in self.best_params.items():
            # Convert value to string for storage
            value_str = json.dumps(param_value) if not isinstance(param_value, str) else param_value

            cursor.execute("""
                INSERT INTO strategy_params (strategy_name, param_name, param_value, is_best)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(strategy_name, param_name) DO UPDATE SET
                    param_value = excluded.param_value,
                    is_best = 1,
                    created_at = CURRENT_TIMESTAMP
            """, (self.strategy_name, param_name, value_str))

        conn.commit()
        conn.close()

    def load_best_params(self) -> Dict[str, Any]:
        """Load best parameters from database."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT param_name, param_value
            FROM strategy_params
            WHERE strategy_name = ? AND is_best = 1
        """, (self.strategy_name,))

        params = {}
        for row in cursor.fetchall():
            param_name = row['param_name']
            param_value_str = row['param_value']

            # Try to parse as JSON, fall back to string
            try:
                params[param_name] = json.loads(param_value_str)
            except (json.JSONDecodeError, TypeError):
                params[param_name] = param_value_str

        conn.close()
        return params

    def promote_params(self, params: Dict[str, Any]):
        """Promote specific parameters as best."""
        self.best_params = params
        self._save_best_params()


class ParameterSpace:
    """Define parameter search space for strategies."""

    @staticmethod
    def suggest_params(trial: Trial, param_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Suggest parameters based on configuration.

        Args:
            trial: Optuna trial
            param_config: Dict of parameter configurations

        Example param_config:
        {
            'rsi_length': {'type': 'int', 'low': 5, 'high': 30},
            'rsi_overbought': {'type': 'int', 'low': 60, 'high': 90},
            'ema_fast': {'type': 'int', 'low': 5, 'high': 50},
            'ema_slow': {'type': 'int', 'low': 20, 'high': 200},
            'stop_loss_pct': {'type': 'float', 'low': 0.01, 'high': 0.05},
            'position_size': {'type': 'categorical', 'choices': ['small', 'medium', 'large']}
        }
        """
        params = {}

        for param_name, config in param_config.items():
            param_type = config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    config['low'],
                    config['high'],
                    step=config.get('step', 1)
                )

            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    config['low'],
                    config['high'],
                    step=config.get('step'),
                    log=config.get('log', False)
                )

            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config['choices']
                )

        return params

    @staticmethod
    def get_default_rsi_config() -> Dict[str, Dict[str, Any]]:
        """Example: RSI strategy parameter space."""
        return {
            'rsi_length': {'type': 'int', 'low': 5, 'high': 30},
            'rsi_oversold': {'type': 'int', 'low': 10, 'high': 40},
            'rsi_overbought': {'type': 'int', 'low': 60, 'high': 90},
            'stop_loss_pct': {'type': 'float', 'low': 0.005, 'high': 0.05, 'log': True},
            'take_profit_pct': {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True}
        }

    @staticmethod
    def get_default_ema_config() -> Dict[str, Dict[str, Any]]:
        """Example: EMA crossover strategy parameter space."""
        return {
            'ema_fast': {'type': 'int', 'low': 5, 'high': 50},
            'ema_slow': {'type': 'int', 'low': 20, 'high': 200},
            'signal_threshold': {'type': 'float', 'low': 0.001, 'high': 0.02},
            'stop_loss_pct': {'type': 'float', 'low': 0.005, 'high': 0.05, 'log': True}
        }


# Example usage and testing
if __name__ == "__main__":
    print("Optuna Optimization Service")
    print("=" * 60)

    # Example objective function
    def example_objective(trial: Trial) -> float:
        """Example objective: optimize simple function."""
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)

        # Objective: minimize (x-2)^2 + (y+3)^2
        # Optimal at x=2, y=-3
        return (x - 2) ** 2 + (y + 3) ** 2

    # Create optimizer
    optimizer = StrategyOptimizer('test_strategy')
    optimizer.create_study(direction='minimize')

    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.optimize(example_objective, n_trials=50)

    print(f"\nBest parameters: {result['best_params']}")
    print(f"Best value: {result['best_value']:.4f}")
    print(f"Expected: x=2.0, y=-3.0")
    print("\nOptimization complete!")
