# backend/ai/reinforcement_learning/agent.py
"""
PPO Trading Agent

Proximal Policy Optimization agent for autonomous trading.
Learns optimal trading strategies through reinforcement learning.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .trading_env import TradingEnvironment, create_env_from_dataframe


class TradingCallback(BaseCallback):
    """
    Callback for tracking training progress and logging metrics.
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called after each step"""
        if self.n_calls % self.check_freq == 0:
            # Get environment
            env = self.training_env.envs[0]

            if hasattr(env, 'get_episode_metrics'):
                metrics = env.get_episode_metrics()

                if self.verbose > 0:
                    print(f"\n[Step {self.n_calls}] Training Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")

        return True


class TradingAgent:
    """
    RL Agent for autonomous trading using PPO algorithm.

    Features:
    - Trains on historical data
    - Learns optimal buy/sell/hold decisions
    - Saves/loads trained models
    - Provides predictions with confidence scores
    """

    def __init__(
        self,
        model_name: str = "trading_agent",
        model_dir: str = "models/rl",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1
    ):
        """
        Initialize trading agent.

        Args:
            model_name: Name for saving/loading model
            model_dir: Directory to save models
            learning_rate: PPO learning rate
            n_steps: Number of steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient for exploration
            verbose: Logging verbosity
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[PPO] = None
        self.env: Optional[TradingEnvironment] = None

        # Hyperparameters
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'verbose': verbose
        }

        # Training history
        self.training_history = []

    def create_model(self, env: TradingEnvironment):
        """
        Create a new PPO model.

        Args:
            env: Trading environment
        """
        self.env = env

        # Wrap environment in Monitor for logging
        monitored_env = Monitor(env)
        vectorized_env = DummyVecEnv([lambda: monitored_env])

        self.model = PPO(
            policy="MlpPolicy",
            env=vectorized_env,
            **self.hyperparameters,
            tensorboard_log=str(self.model_dir / "tensorboard")
        )

        print(f"[INFO] Created new PPO model: {self.model_name}")

    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[TradingEnvironment] = None,
        eval_freq: int = 10000,
        save_freq: int = 10000
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            total_timesteps: Total training steps
            eval_env: Optional evaluation environment
            eval_freq: Evaluation frequency
            save_freq: Model save frequency

        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")

        # Setup callbacks
        callbacks = [TradingCallback(check_freq=1000, verbose=1)]

        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            eval_callback = EvalCallback(
                Monitor(eval_env),
                best_model_save_path=str(self.model_dir / "best_model"),
                log_path=str(self.model_dir / "eval_logs"),
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # Train
        start_time = datetime.now()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # Save model
        self.save()

        # Get training metrics
        metrics = {
            'total_timesteps': total_timesteps,
            'training_time_seconds': training_time,
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters,
            'trained_at': datetime.utcnow().isoformat()
        }

        self.training_history.append(metrics)

        print(f"\n{'='*60}")
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Model saved to: {self.model_dir / self.model_name}")
        print(f"{'='*60}\n")

        return metrics

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, float]:
        """
        Predict action for given observation.

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, confidence)
            - action: 0 (HOLD), 1 (BUY), or 2 (SELL)
            - confidence: Model's confidence in action (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or load() first.")

        action, _states = self.model.predict(observation, deterministic=deterministic)

        # Get action probabilities for confidence
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]

        confidence = float(probs[int(action)])

        return int(action), confidence

    def predict_with_reasoning(
        self,
        observation: np.ndarray,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict action with detailed reasoning.

        Args:
            observation: Current state observation
            market_context: Optional market context for explanation

        Returns:
            Dict with action, confidence, and reasoning
        """
        action, confidence = self.predict(observation)

        # Map action to string
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_str = action_map[action]

        # Get all action probabilities
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]

        reasoning = {
            'action': action_str,
            'action_id': int(action),
            'confidence': confidence,
            'all_action_probs': {
                'HOLD': float(probs[0]),
                'BUY': float(probs[1]),
                'SELL': float(probs[2])
            },
            'model_name': self.model_name,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add market context if provided
        if market_context:
            reasoning['market_context'] = market_context

        return reasoning

    def save(self, custom_path: Optional[str] = None):
        """
        Save model to disk.

        Args:
            custom_path: Optional custom save path
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = custom_path or str(self.model_dir / self.model_name)
        self.model.save(save_path)

        # Save hyperparameters and training history
        metadata = {
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters,
            'training_history': self.training_history,
            'saved_at': datetime.utcnow().isoformat()
        }

        metadata_path = f"{save_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Model saved to: {save_path}")
        print(f"[INFO] Metadata saved to: {metadata_path}")

    def load(self, custom_path: Optional[str] = None):
        """
        Load model from disk.

        Args:
            custom_path: Optional custom load path
        """
        load_path = custom_path or str(self.model_dir / self.model_name)

        if not os.path.exists(f"{load_path}.zip"):
            raise FileNotFoundError(f"Model not found at: {load_path}")

        self.model = PPO.load(load_path)

        # Load metadata if exists
        metadata_path = f"{load_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_history = metadata.get('training_history', [])
                print(f"[INFO] Loaded model: {metadata.get('model_name')}")
                print(f"[INFO] Last trained: {metadata.get('saved_at')}")
        else:
            print(f"[INFO] Model loaded from: {load_path}")

    def evaluate(
        self,
        eval_env: TradingEnvironment,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent performance.

        Args:
            eval_env: Environment for evaluation
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        episode_rewards = []
        episode_metrics = []

        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(int(action))
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_metrics.append(eval_env.get_episode_metrics())

        # Aggregate metrics
        avg_metrics = {}
        for key in episode_metrics[0].keys():
            values = [m[key] for m in episode_metrics]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
            avg_metrics[f'min_{key}'] = np.min(values)
            avg_metrics[f'max_{key}'] = np.max(values)

        return {
            'n_episodes': n_episodes,
            'episode_rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            **avg_metrics
        }


def train_agent(
    df,
    model_name: str = "trading_agent_v1",
    total_timesteps: int = 100000,
    **kwargs
) -> TradingAgent:
    """
    Convenience function to train a new agent.

    Args:
        df: DataFrame with OHLCV and features
        model_name: Name for the model
        total_timesteps: Training duration
        **kwargs: Additional arguments for TradingAgent

    Returns:
        Trained TradingAgent
    """
    # Create environment
    env = create_env_from_dataframe(df)

    # Create agent
    agent = TradingAgent(model_name=model_name, **kwargs)
    agent.create_model(env)

    # Train
    agent.train(total_timesteps=total_timesteps)

    return agent


def load_trained_agent(model_path: str) -> TradingAgent:
    """
    Load a pre-trained agent.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded TradingAgent
    """
    agent = TradingAgent()
    agent.load(model_path)
    return agent
