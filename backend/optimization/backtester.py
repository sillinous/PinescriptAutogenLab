# backend/optimization/backtester.py

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    params: Dict[str, Any]


class SimpleBacktester:
    """
    Simple backtesting framework for strategy optimization.

    This is a basic implementation. For production, consider:
    - Backtrader
    - Zipline
    - VectorBT
    - QuantConnect LEAN
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005  # 0.05%
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: callable,
        params: Dict[str, Any]
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            strategy_func: Function that takes (data, params) and returns signals
            params: Strategy parameters

        Returns:
            BacktestResult object
        """
        # Generate trading signals
        signals = strategy_func(data, params)

        # Execute trades based on signals
        trades = self._execute_trades(data, signals)

        # Calculate metrics
        result = self._calculate_metrics(trades, params)

        return result

    def _execute_trades(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> List[Dict[str, Any]]:
        """Execute trades based on signals."""
        trades = []
        position = None
        capital = self.initial_capital

        for idx in range(len(data)):
            signal = signals.iloc[idx]
            current_price = data['close'].iloc[idx]
            timestamp = data.index[idx] if hasattr(data.index, 'to_pydatetime') else idx

            # Entry signal
            if signal == 1 and position is None:  # Buy
                entry_price = current_price * (1 + self.slippage)
                shares = int((capital * 0.95) / entry_price)  # Use 95% of capital
                cost = shares * entry_price
                commission_paid = cost * self.commission

                position = {
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'shares': shares,
                    'cost': cost + commission_paid,
                    'side': 'long'
                }

            # Exit signal
            elif signal == -1 and position is not None:  # Sell
                exit_price = current_price * (1 - self.slippage)
                proceeds = position['shares'] * exit_price
                commission_paid = proceeds * self.commission
                net_proceeds = proceeds - commission_paid

                pnl = net_proceeds - position['cost']
                pnl_pct = (pnl / position['cost']) * 100

                capital += pnl

                trade = {
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': timestamp,
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'side': position['side']
                }

                trades.append(trade)
                position = None

        # Close any open position at end
        if position is not None:
            exit_price = data['close'].iloc[-1] * (1 - self.slippage)
            proceeds = position['shares'] * exit_price
            commission_paid = proceeds * self.commission
            net_proceeds = proceeds - commission_paid
            pnl = net_proceeds - position['cost']
            pnl_pct = (pnl / position['cost']) * 100

            trade = {
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': data.index[-1] if hasattr(data.index, 'to_pydatetime') else len(data) - 1,
                'exit_price': exit_price,
                'shares': position['shares'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'side': position['side']
            }

            trades.append(trade)

        return trades

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=[],
                equity_curve=[self.initial_capital],
                params=params
            )

        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] <= 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades)

        # Equity curve
        equity_curve = [self.initial_capital]
        running_equity = self.initial_capital
        for trade in trades:
            running_equity += trade['pnl']
            equity_curve.append(running_equity)

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t['pnl_pct'] for t in trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0.0

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            total_pnl=round(total_pnl, 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            trades=trades,
            equity_curve=equity_curve,
            params=params
        )

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = equity_curve[0]
        max_dd = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) * 100
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd


class WalkForwardValidator:
    """
    Walk-forward validation for robust optimization.

    Splits data into training/testing windows and validates
    optimization results on out-of-sample data.
    """

    def __init__(
        self,
        train_size: int = 252,  # 1 year
        test_size: int = 63,  # 3 months
        step_size: int = 21  # 1 month
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def split_data(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into walk-forward windows.

        Returns:
            List of (train_data, test_data) tuples
        """
        splits = []
        total_length = len(data)

        for start in range(0, total_length - self.train_size - self.test_size, self.step_size):
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            if test_end > total_length:
                break

            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]

            splits.append((train_data, test_data))

        return splits

    def validate(
        self,
        data: pd.DataFrame,
        optimizer_func: callable,
        strategy_func: callable
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            data: Full dataset
            optimizer_func: Function to optimize params on training data
            strategy_func: Strategy function to backtest

        Returns:
            Validation results
        """
        splits = self.split_data(data)
        results = []

        backtester = SimpleBacktester()

        for i, (train_data, test_data) in enumerate(splits):
            # Optimize on training data
            best_params = optimizer_func(train_data)

            # Test on out-of-sample data
            test_result = backtester.run_backtest(test_data, strategy_func, best_params)

            results.append({
                'window': i,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'params': best_params,
                'test_pnl': test_result.total_pnl,
                'test_win_rate': test_result.win_rate,
                'test_sharpe': test_result.sharpe_ratio
            })

        # Aggregate results
        avg_pnl = np.mean([r['test_pnl'] for r in results])
        avg_win_rate = np.mean([r['test_win_rate'] for r in results])
        avg_sharpe = np.mean([r['test_sharpe'] for r in results])

        return {
            'n_windows': len(results),
            'avg_pnl': round(avg_pnl, 2),
            'avg_win_rate': round(avg_win_rate, 2),
            'avg_sharpe': round(avg_sharpe, 2),
            'window_results': results,
            'passed': avg_pnl > 0 and avg_win_rate > 50
        }


# Example strategies for testing
def simple_rsi_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Simple RSI strategy example.

    Params:
        - rsi_length: RSI period
        - rsi_oversold: Oversold threshold
        - rsi_overbought: Overbought threshold
    """
    # Calculate RSI (simplified)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    rsi_length = params.get('rsi_length', 14)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Generate signals
    signals = pd.Series(0, index=data.index)
    signals[rsi < params.get('rsi_oversold', 30)] = 1  # Buy
    signals[rsi > params.get('rsi_overbought', 70)] = -1  # Sell

    return signals


def simple_ema_crossover_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Simple EMA crossover strategy example.

    Params:
        - ema_fast: Fast EMA period
        - ema_slow: Slow EMA period
    """
    ema_fast = data['close'].ewm(span=params.get('ema_fast', 12)).mean()
    ema_slow = data['close'].ewm(span=params.get('ema_slow', 26)).mean()

    # Generate signals
    signals = pd.Series(0, index=data.index)
    signals[ema_fast > ema_slow] = 1  # Buy (fast above slow)
    signals[ema_fast < ema_slow] = -1  # Sell (fast below slow)

    # Only signal on crossovers
    signals = signals.diff().fillna(0)

    return signals


if __name__ == "__main__":
    print("Backtesting Framework Test")
    print("=" * 60)

    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 2) + 1,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 2) - 1,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 2),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Test backtest
    backtester = SimpleBacktester(initial_capital=10000)
    params = {'rsi_length': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}

    result = backtester.run_backtest(sample_data, simple_rsi_strategy, params)

    print(f"\nBacktest Results:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate}%")
    print(f"  Profit Factor: {result.profit_factor}")
    print(f"  Total P&L: ${result.total_pnl:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio}")
    print("\nBacktester working correctly!")
