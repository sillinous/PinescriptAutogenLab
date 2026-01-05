# backend/ai/reinforcement_learning/trading_env.py
"""
Trading Environment for Reinforcement Learning

Gymnasium (Gym) compatible environment for training RL agents to trade.
The agent learns optimal buy/sell/hold decisions through rewards.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from enum import IntEnum


class Actions(IntEnum):
    """Possible trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for RL agents.

    Observation Space:
        - Market features (price, indicators, etc.)
        - Position information (holdings, cash, P&L)

    Action Space:
        - 0: HOLD (do nothing)
        - 1: BUY (enter long position)
        - 2: SELL (exit position or short)

    Reward:
        - Profit/loss from trades
        - Sharpe ratio improvement
        - Penalty for excessive trading (transaction costs)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1%
        lookback_window: int = 60,
        reward_scaling: float = 1e-4,
        features: Optional[list] = None
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV data and features
            initial_balance: Starting cash balance
            transaction_cost: Trading fee as fraction (0.001 = 0.1%)
            lookback_window: Number of past bars to include in observation
            reward_scaling: Scale rewards to reasonable range
            features: List of feature column names (None = all numeric)
        """
        super().__init__()

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling

        # Get feature columns
        if features is None:
            # Use all numeric columns except OHLCV
            self.features = [col for col in df.select_dtypes(include=[np.number]).columns
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        else:
            self.features = features

        self.num_features = len(self.features)

        # Ensure we have enough data
        if len(df) < lookback_window + 1:
            raise ValueError(f"DataFrame too short. Need at least {lookback_window + 1} rows.")

        # Action space: HOLD, BUY, SELL
        self.action_space = spaces.Discrete(3)

        # Observation space: [market features + position info]
        # Market features: lookback_window x num_features
        # Position info: cash, holdings, avg_buy_price, total_value, pnl
        market_obs_shape = lookback_window * self.num_features
        position_obs_shape = 5

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(market_obs_shape + position_obs_shape,),
            dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.max_steps = len(df) - lookback_window

        # Account state
        self.balance = initial_balance
        self.shares_held = 0.0
        self.avg_buy_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        # Episode metrics
        self.episode_reward = 0.0
        self.episode_trades = 0
        self.max_account_value = initial_balance
        self.min_account_value = initial_balance

        # History for rendering/analysis
        self.account_history = []
        self.trades_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.avg_buy_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        self.episode_reward = 0.0
        self.episode_trades = 0
        self.max_account_value = self.initial_balance
        self.min_account_value = self.initial_balance

        self.account_history = []
        self.trades_history = []

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return next state.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Get current price
        current_idx = self.lookback_window + self.current_step
        current_price = self.df.iloc[current_idx]['close']

        # Execute action
        reward = 0.0
        action_taken = Actions(action).name

        if action == Actions.BUY and self.shares_held == 0:
            # Buy: use all available cash
            shares_to_buy = self.balance / (current_price * (1 + self.transaction_cost))
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)

            if cost <= self.balance:
                self.shares_held = shares_to_buy
                self.avg_buy_price = current_price
                self.balance -= cost
                self.total_trades += 1
                self.episode_trades += 1

                self.trades_history.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'cost': cost
                })

        elif action == Actions.SELL and self.shares_held > 0:
            # Sell: sell all shares
            proceeds = self.shares_held * current_price * (1 - self.transaction_cost)
            pnl = proceeds - (self.shares_held * self.avg_buy_price)

            self.balance += proceeds
            reward = pnl * self.reward_scaling  # Scale reward

            if pnl > 0:
                self.winning_trades += 1

            self.trades_history.append({
                'step': self.current_step,
                'action': 'SELL',
                'price': current_price,
                'shares': self.shares_held,
                'proceeds': proceeds,
                'pnl': pnl
            })

            self.shares_held = 0.0
            self.avg_buy_price = 0.0
            self.total_trades += 1
            self.episode_trades += 1

        # Calculate account value
        account_value = self.balance + (self.shares_held * current_price)

        # Track account value extremes
        self.max_account_value = max(self.max_account_value, account_value)
        self.min_account_value = min(self.min_account_value, account_value)

        # Additional reward shaping
        # Reward for account value increase
        value_change = account_value - self.initial_balance
        reward += value_change * self.reward_scaling * 0.01

        # Penalty for excessive trading
        if action != Actions.HOLD:
            reward -= 0.001  # Small penalty for each trade

        # Penalty for drawdown
        if account_value < self.initial_balance:
            drawdown_penalty = (self.initial_balance - account_value) / self.initial_balance
            reward -= drawdown_penalty * 0.1

        self.episode_reward += reward

        # Record history
        self.account_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'account_value': account_value,
            'action': action_taken
        })

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_observation()

        # Info dict
        info = {
            'account_value': account_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'episode_reward': self.episode_reward
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Flattened array of [market_features, position_info]
        """
        current_idx = self.lookback_window + self.current_step

        # Get market features for lookback window
        market_data = self.df.iloc[current_idx - self.lookback_window:current_idx][self.features].values

        # Handle NaN and inf
        market_data = np.nan_to_num(market_data, nan=0.0, posinf=1e6, neginf=-1e6)

        # Flatten market features
        market_obs = market_data.flatten()

        # Get current price for position value
        current_price = self.df.iloc[current_idx]['close']
        account_value = self.balance + (self.shares_held * current_price)

        # Position information (normalized)
        position_obs = np.array([
            self.balance / self.initial_balance,  # Cash ratio
            self.shares_held * current_price / self.initial_balance,  # Position value ratio
            (current_price - self.avg_buy_price) / (self.avg_buy_price + 1e-8) if self.avg_buy_price > 0 else 0,  # Unrealized P&L
            account_value / self.initial_balance,  # Total account value ratio
            self.total_trades / 100.0  # Trade count (scaled)
        ], dtype=np.float32)

        # Combine observations
        obs = np.concatenate([market_obs, position_obs])

        return obs.astype(np.float32)

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            current_idx = self.lookback_window + self.current_step
            current_price = self.df.iloc[current_idx]['close'] if current_idx < len(self.df) else 0
            account_value = self.balance + (self.shares_held * current_price)

            print(f"\n=== Step {self.current_step} ===")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares Held: {self.shares_held:.4f}")
            print(f"Account Value: ${account_value:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Episode Reward: {self.episode_reward:.4f}")

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get episode performance metrics.

        Returns:
            Dict with Sharpe ratio, win rate, max drawdown, etc.
        """
        if not self.account_history:
            return {}

        account_values = [h['account_value'] for h in self.account_history]

        # Calculate returns
        returns = np.diff(account_values) / account_values[:-1]

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(account_values)
        drawdown = (peak - account_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Total P&L
        total_pnl = account_values[-1] - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        return {
            'final_account_value': account_values[-1],
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'episode_reward': self.episode_reward
        }

    def close(self):
        """Clean up resources."""
        pass


def create_env_from_dataframe(
    df: pd.DataFrame,
    **kwargs
) -> TradingEnvironment:
    """
    Factory function to create trading environment from DataFrame.

    Args:
        df: DataFrame with OHLCV and features
        **kwargs: Additional arguments for TradingEnvironment

    Returns:
        TradingEnvironment instance
    """
    return TradingEnvironment(df, **kwargs)
