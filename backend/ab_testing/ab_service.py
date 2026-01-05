# backend/ab_testing/ab_service.py

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from backend.database import get_db
import json
import sqlite3
from scipy import stats


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_name: str
    variant_a_params: Dict[str, Any]  # Control
    variant_b_params: Dict[str, Any]  # Candidate
    start_time: datetime
    end_time: Optional[datetime] = None
    min_sample_size: int = 30  # Minimum trades per variant
    significance_level: float = 0.05  # p-value threshold
    status: str = "running"  # running, completed, promoted


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    test_name: str
    variant_a_trades: int
    variant_b_trades: int
    variant_a_win_rate: float
    variant_b_win_rate: float
    variant_a_avg_pnl: float
    variant_b_avg_pnl: float
    variant_a_sharpe: float
    variant_b_sharpe: float
    p_value: float
    is_significant: bool
    winner: str  # 'A', 'B', or 'inconclusive'
    confidence: float


class ABTestingService:
    """
    A/B testing service for strategy comparison.

    Allows shadow deployment of candidate strategies alongside
    control strategies to measure statistical significance.
    """

    def __init__(self):
        self._init_db_tables()

    def _init_db_tables(self):
        """Initialize A/B testing tables if they don't exist."""
        conn = get_db()
        cursor = conn.cursor()

        # A/B test configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT UNIQUE NOT NULL,
                variant_a_params TEXT NOT NULL,
                variant_b_params TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                min_sample_size INTEGER DEFAULT 30,
                significance_level REAL DEFAULT 0.05,
                status TEXT DEFAULT 'running',
                winner TEXT,
                p_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # A/B test trade results (shadow trades)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                variant TEXT NOT NULL,  -- 'A' or 'B'
                signal_time TIMESTAMP NOT NULL,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                is_winner BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_name) REFERENCES ab_tests(test_name)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ab_test_trades_test ON ab_test_trades(test_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ab_test_trades_variant ON ab_test_trades(variant)")

        conn.commit()
        conn.close()

    def create_test(
        self,
        test_name: str,
        variant_a_params: Dict[str, Any],
        variant_b_params: Dict[str, Any],
        min_sample_size: int = 30,
        significance_level: float = 0.05
    ) -> ABTestConfig:
        """
        Create new A/B test.

        Args:
            test_name: Unique name for the test
            variant_a_params: Control strategy parameters
            variant_b_params: Candidate strategy parameters
            min_sample_size: Minimum trades before evaluation
            significance_level: Statistical significance threshold

        Returns:
            ABTestConfig object
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ab_tests (
                test_name, variant_a_params, variant_b_params,
                min_sample_size, significance_level
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            test_name,
            json.dumps(variant_a_params),
            json.dumps(variant_b_params),
            min_sample_size,
            significance_level
        ))

        conn.commit()
        conn.close()

        return ABTestConfig(
            test_name=test_name,
            variant_a_params=variant_a_params,
            variant_b_params=variant_b_params,
            start_time=datetime.now(),
            min_sample_size=min_sample_size,
            significance_level=significance_level
        )

    def record_trade(
        self,
        test_name: str,
        variant: str,
        signal_time: datetime,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None
    ):
        """
        Record a shadow trade for A/B testing.

        This is called when a signal is generated to log what
        WOULD have happened with each variant's parameters.
        """
        conn = get_db()
        cursor = conn.cursor()

        # Convert to Python bool to avoid numpy.bool_ JSON serialization issues
        is_winner = bool(pnl > 0) if pnl is not None else None

        cursor.execute("""
            INSERT INTO ab_test_trades (
                test_name, variant, signal_time, symbol, side,
                entry_price, exit_price, pnl, pnl_pct, is_winner
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_name, variant, signal_time, symbol, side,
            entry_price, exit_price, pnl, pnl_pct, is_winner
        ))

        conn.commit()
        conn.close()

    def get_test_results(self, test_name: str) -> ABTestResult:
        """
        Get current results for an A/B test.

        Performs statistical analysis to determine if there's
        a significant difference between variants.
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get variant A trades
        cursor.execute("""
            SELECT pnl, pnl_pct, is_winner
            FROM ab_test_trades
            WHERE test_name = ? AND variant = 'A' AND pnl IS NOT NULL
        """, (test_name,))
        variant_a_trades = cursor.fetchall()

        # Get variant B trades
        cursor.execute("""
            SELECT pnl, pnl_pct, is_winner
            FROM ab_test_trades
            WHERE test_name = ? AND variant = 'B' AND pnl IS NOT NULL
        """, (test_name,))
        variant_b_trades = cursor.fetchall()

        conn.close()

        # Calculate metrics
        a_pnls = [float(t['pnl']) for t in variant_a_trades]
        b_pnls = [float(t['pnl']) for t in variant_b_trades]

        a_wins = sum(1 for t in variant_a_trades if t['is_winner'])
        b_wins = sum(1 for t in variant_b_trades if t['is_winner'])

        a_win_rate = (a_wins / len(variant_a_trades) * 100) if variant_a_trades else 0
        b_win_rate = (b_wins / len(variant_b_trades) * 100) if variant_b_trades else 0

        a_avg_pnl = sum(a_pnls) / len(a_pnls) if a_pnls else 0
        b_avg_pnl = sum(b_pnls) / len(b_pnls) if b_pnls else 0

        a_sharpe = self._calculate_sharpe(a_pnls) if len(a_pnls) > 1 else 0
        b_sharpe = self._calculate_sharpe(b_pnls) if len(b_pnls) > 1 else 0

        # Statistical test (t-test for PnL difference)
        if len(a_pnls) >= 10 and len(b_pnls) >= 10:
            t_stat, p_value = stats.ttest_ind(a_pnls, b_pnls)
            p_value = float(p_value)  # Convert numpy float to Python float
            is_significant = bool(p_value < 0.05)  # Convert to Python bool
        else:
            p_value = 1.0
            is_significant = False

        # Determine winner
        if is_significant:
            winner = 'B' if b_avg_pnl > a_avg_pnl else 'A'
            confidence = (1 - p_value) * 100
        else:
            winner = 'inconclusive'
            confidence = 0

        return ABTestResult(
            test_name=test_name,
            variant_a_trades=len(variant_a_trades),
            variant_b_trades=len(variant_b_trades),
            variant_a_win_rate=round(a_win_rate, 2),
            variant_b_win_rate=round(b_win_rate, 2),
            variant_a_avg_pnl=round(a_avg_pnl, 2),
            variant_b_avg_pnl=round(b_avg_pnl, 2),
            variant_a_sharpe=round(a_sharpe, 2),
            variant_b_sharpe=round(b_sharpe, 2),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            winner=winner,
            confidence=round(confidence, 2)
        )

    def promote_winner(self, test_name: str):
        """
        Promote the winning variant to production.

        This marks the test as complete and promotes the better-performing
        variant's parameters as the new best parameters.
        """
        results = self.get_test_results(test_name)

        if results.winner == 'inconclusive':
            raise ValueError("Cannot promote: test results inconclusive")

        # Get test config
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT variant_a_params, variant_b_params
            FROM ab_tests
            WHERE test_name = ?
        """, (test_name,))

        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Test not found: {test_name}")

        # Get winning params
        if results.winner == 'A':
            winning_params = json.loads(row['variant_a_params'])
        else:
            winning_params = json.loads(row['variant_b_params'])

        # Update test status
        cursor.execute("""
            UPDATE ab_tests
            SET status = 'promoted',
                winner = ?,
                p_value = ?,
                end_time = CURRENT_TIMESTAMP
            WHERE test_name = ?
        """, (results.winner, results.p_value, test_name))

        # Promote params (integrate with optimizer)
        from backend.optimization.integrated_optimizer import IntegratedStrategyOptimizer
        optimizer = IntegratedStrategyOptimizer(test_name)
        optimizer.promote_params(winning_params)

        conn.commit()
        conn.close()

        return {
            'winner': results.winner,
            'promoted_params': winning_params,
            'confidence': results.confidence
        }

    def _calculate_sharpe(self, pnls: List[float]) -> float:
        """Calculate Sharpe ratio from P&L list."""
        if len(pnls) < 2:
            return 0.0

        import numpy as np
        returns = np.array(pnls)
        if np.std(returns) == 0:
            return 0.0

        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        return float(sharpe)  # Convert numpy float to Python float

    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get all active A/B tests."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM ab_tests
            WHERE status = 'running'
            ORDER BY created_at DESC
        """)

        tests = []
        for row in cursor.fetchall():
            tests.append({
                'test_name': row['test_name'],
                'variant_a_params': json.loads(row['variant_a_params']),
                'variant_b_params': json.loads(row['variant_b_params']),
                'start_time': row['start_time'],
                'status': row['status']
            })

        conn.close()
        return tests


# Global A/B testing service
_ab_service: Optional[ABTestingService] = None


def get_ab_service() -> ABTestingService:
    """Get A/B testing service instance."""
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestingService()
    return _ab_service
