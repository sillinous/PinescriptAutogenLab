# backend/optimization/integrated_optimizer.py

from typing import Dict, Any, Optional, Callable
from optuna.trial import Trial
import pandas as pd
import numpy as np
from backend.optimization.optuna_service import StrategyOptimizer, ParameterSpace
from backend.optimization.backtester import SimpleBacktester, BacktestResult, simple_rsi_strategy, simple_ema_crossover_strategy
import json
from pathlib import Path


class IntegratedStrategyOptimizer:
    """
    Combines Optuna optimization with backtesting.

    This is the main service used by the API.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.optimizer = StrategyOptimizer(strategy_name)
        self.backtester = SimpleBacktester()
        self.current_progress = 0
        self.total_trials = 0

    def optimize_strategy(
        self,
        data: pd.DataFrame,
        param_config: Dict[str, Dict[str, Any]],
        strategy_func: Callable,
        n_trials: int = 100,
        direction: str = "maximize",
        metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data.

        Args:
            data: Historical OHLCV data
            param_config: Parameter space configuration
            strategy_func: Strategy signal generation function
            n_trials: Number of optimization trials
            direction: 'maximize' or 'minimize'
            metric: Metric to optimize ('sharpe_ratio', 'total_pnl', 'win_rate', etc.)

        Returns:
            Optimization results with best parameters
        """
        self.total_trials = n_trials
        self.current_progress = 0

        # Create objective function
        def objective(trial: Trial) -> float:
            # Suggest parameters
            params = ParameterSpace.suggest_params(trial, param_config)

            # Run backtest
            try:
                result = self.backtester.run_backtest(data, strategy_func, params)

                # Update progress
                self.current_progress = trial.number + 1

                # Return metric to optimize
                if metric == 'sharpe_ratio':
                    return result.sharpe_ratio
                elif metric == 'total_pnl':
                    return result.total_pnl
                elif metric == 'win_rate':
                    return result.win_rate
                elif metric == 'profit_factor':
                    return result.profit_factor
                else:
                    return result.sharpe_ratio  # Default

            except Exception as e:
                print(f"[ERROR] Trial {trial.number} failed: {e}")
                return float('-inf') if direction == 'maximize' else float('inf')

        # Create study
        self.optimizer.create_study(direction=direction)

        # Run optimization
        result = self.optimizer.optimize(
            objective,
            n_trials=n_trials,
            show_progress=True
        )

        # Get parameter importance
        param_importance = self.optimizer.get_param_importance()

        # Run final backtest with best params
        best_backtest = self.backtester.run_backtest(
            data,
            strategy_func,
            result['best_params']
        )

        return {
            **result,
            'param_importance': param_importance,
            'final_backtest': {
                'total_trades': best_backtest.total_trades,
                'win_rate': best_backtest.win_rate,
                'profit_factor': best_backtest.profit_factor,
                'total_pnl': best_backtest.total_pnl,
                'max_drawdown': best_backtest.max_drawdown,
                'sharpe_ratio': best_backtest.sharpe_ratio
            }
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get current optimization progress."""
        progress_pct = (self.current_progress / self.total_trials * 100) if self.total_trials > 0 else 0

        return {
            'progress': round(progress_pct, 2),
            'current_trial': self.current_progress,
            'total_trials': self.total_trials,
            'best_parameters': self.optimizer.best_params,
            'status': 'running' if self.current_progress < self.total_trials else 'completed'
        }

    def load_best_params(self) -> Dict[str, Any]:
        """Load previously optimized parameters."""
        return self.optimizer.load_best_params()

    def promote_params(self, params: Dict[str, Any]):
        """Promote parameters as best."""
        self.optimizer.promote_params(params)


# Pre-configured strategies
STRATEGY_CONFIGS = {
    'rsi': {
        'name': 'RSI Strategy',
        'param_config': ParameterSpace.get_default_rsi_config(),
        'strategy_func': simple_rsi_strategy,
        'metric': 'sharpe_ratio'
    },
    'ema_crossover': {
        'name': 'EMA Crossover',
        'param_config': ParameterSpace.get_default_ema_config(),
        'strategy_func': simple_ema_crossover_strategy,
        'metric': 'sharpe_ratio'
    }
}


def get_strategy_config(strategy_type: str) -> Optional[Dict[str, Any]]:
    """Get pre-configured strategy settings."""
    return STRATEGY_CONFIGS.get(strategy_type)


# Global optimization state (for API)
_active_optimizations: Dict[str, IntegratedStrategyOptimizer] = {}


def start_optimization(
    strategy_name: str,
    strategy_type: str,
    data: pd.DataFrame,
    n_trials: int = 100
) -> str:
    """Start optimization in background (called by API)."""
    global _active_optimizations

    config = get_strategy_config(strategy_type)
    if not config:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    optimizer = IntegratedStrategyOptimizer(strategy_name)
    _active_optimizations[strategy_name] = optimizer

    # Start optimization (in real implementation, run in background thread/celery)
    result = optimizer.optimize_strategy(
        data=data,
        param_config=config['param_config'],
        strategy_func=config['strategy_func'],
        n_trials=n_trials,
        metric=config['metric']
    )

    return strategy_name


def get_optimization_status(strategy_name: str) -> Optional[Dict[str, Any]]:
    """Get optimization status."""
    global _active_optimizations

    optimizer = _active_optimizations.get(strategy_name)
    if not optimizer:
        # Try to load from disk
        optimizer = IntegratedStrategyOptimizer(strategy_name)
        best_params = optimizer.load_best_params()

        if best_params:
            return {
                'progress': 100,
                'status': 'completed',
                'best_parameters': best_params
            }
        return None

    return optimizer.get_progress()
