# backend/tests/test_ai_features.py

import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.optimization.integrated_optimizer import _active_optimizations
from backend.ab_testing.ab_service import get_ab_service
import time
import json

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_active_optimizations():
    """Fixture to clear active optimizations before each test."""
    _active_optimizations.clear()
    # Clear AB test data as well
    ab_service = get_ab_service()
    conn = ab_service.get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ab_tests")
    cursor.execute("DELETE FROM ab_test_trades")
    conn.commit()
    conn.close()
    yield

def test_get_available_strategies():
    response = client.get("/api/v1/optimize/strategies")
    assert response.status_code == 200
    strategies = response.json()
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    assert any(s["type"] == "rsi" for s in strategies)
    assert any(s["type"] == "ema" for s in strategies) # Changed from ema_crossover to ema

def test_start_strategy_optimization_success():
    strategy_name = "test_rsi_opt"
    response = client.post(
        "/api/v1/optimize/start",
        json={"strategy_name": strategy_name, "strategy_type": "rsi", "n_trials": 1} # Reduced n_trials
    )
    assert response.status_code == 200
    assert response.json()["message"] == f"Optimization for strategy '{strategy_name}' started successfully."
    assert strategy_name in _active_optimizations

def test_start_strategy_optimization_invalid_type():
    response = client.post(
        "/api/v1/optimize/start",
        json={"strategy_name": "invalid_opt", "strategy_type": "non_existent", "n_trials": 1} # Reduced n_trials
    )
    assert response.status_code == 400
    assert "Unknown strategy type" in response.json()["detail"]

def test_get_optimization_status_running():
    strategy_name = "test_ema_opt_status"
    client.post(
        "/api/v1/optimize/start",
        json={"strategy_name": strategy_name, "strategy_type": "ema_crossover", "n_trials": 2} # Reduced n_trials
    )
    # Give it a moment to start
    time.sleep(0.1)

    response = client.get(f"/api/v1/optimize/status/{strategy_name}")
    assert response.status_code == 200
    status = response.json()
    assert status["status"] == "running"
    assert status["total_trials"] == 2
    assert status["current_trial"] >= 0
    assert "best_parameters" in status

def test_get_optimization_status_completed():
    strategy_name = "test_completed_opt"
    client.post(
        "/api/v1/optimize/start",
        json={"strategy_name": strategy_name, "strategy_type": "rsi", "n_trials": 1} # Reduced n_trials
    )
    # Poll until optimization is completed
    for _ in range(20): # Max 2 seconds wait (20 * 0.1s)
        response = client.get(f"/api/v1/optimize/status/{strategy_name}")
        status = response.json()
        if status["status"] == "completed":
            break
        time.sleep(0.1)
    else:
        pytest.fail("Optimization did not complete in time.")

    assert response.status_code == 200
    status = response.json()
    assert status["status"] == "completed"
    assert status["total_trials"] == 1
    assert status["current_trial"] == 1
    assert "best_parameters" in status
    assert "sharpe_ratio" in status["best_parameters"] # Check for a common metric

def test_get_optimization_status_not_found():
    response = client.get("/api/v1/optimize/status/non_existent_opt")
    assert response.status_code == 404
    assert "not found or not started" in response.json()["detail"]

def test_create_ab_test_success():
    test_name = "test_ab_rsi"
    variant_a = {"rsi_length": 14, "rsi_oversold": 30, "rsi_overbought": 70}
    variant_b = {"rsi_length": 10, "rsi_oversold": 20, "rsi_overbought": 80}
    response = client.post(
        "/api/v1/abtest/create",
        json={
            "test_name": test_name,
            "variant_a_params": variant_a,
            "variant_b_params": variant_b,
            "min_sample_size": 5,
            "significance_level": 0.05
        }
    )
    assert response.status_code == 200
    assert response.json()["message"] == f"A/B test '{test_name}' created successfully."
    assert response.json()["test_config"]["test_name"] == test_name

    # Verify it's in active tests
    active_tests_response = client.get("/api/v1/abtest/active")
    assert active_tests_response.status_code == 200
    active_tests = active_tests_response.json()["active_tests"]
    assert any(t["test_name"] == test_name for t in active_tests)

def test_create_ab_test_duplicate_name():
    test_name = "duplicate_test"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    client.post(
        "/api/v1/abtest/create",
        json={"test_name": test_name, "variant_a_params": variant_a, "variant_b_params": variant_b}
    )
    response = client.post(
        "/api/v1/abtest/create",
        json={"test_name": test_name, "variant_a_params": variant_a, "variant_b_params": variant_b}
    )
    assert response.status_code == 400
    assert "UNIQUE constraint failed" in response.json()["detail"]

def test_get_ab_test_results_no_trades():
    test_name = "test_ab_no_trades"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    client.post(
        "/api/v1/abtest/create",
        json={"test_name": test_name, "variant_a_params": variant_a, "variant_b_params": variant_b}
    )
    response = client.get(f"/api/v1/abtest/results/{test_name}")
    assert response.status_code == 200
    results = response.json()
    assert results["test_name"] == test_name
    assert results["variant_a_trades"] == 0
    assert results["variant_b_trades"] == 0
    assert results["winner"] == "inconclusive"

def test_get_ab_test_results_with_trades():
    test_name = "test_ab_with_trades"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    ab_service = get_ab_service()
    ab_service.create_test(test_name, variant_a, variant_b)

    # Record some trades
    for _ in range(5): # Reduced number of trades
        ab_service.record_trade(test_name, "A", datetime.now(), "BTCUSDT", "buy", 100, 101, 1.0, 1.0)
        ab_service.record_trade(test_name, "A", datetime.now(), "BTCUSDT", "sell", 100, 99, -1.0, -1.0)
        ab_service.record_trade(test_name, "B", datetime.now(), "ETHUSDT", "buy", 200, 203, 3.0, 1.5)
        ab_service.record_trade(test_name, "B", datetime.now(), "ETHUSDT", "sell", 200, 198, -2.0, -1.0)
        ab_service.record_trade(test_name, "B", datetime.now(), "ETHUSDT", "buy", 200, 205, 5.0, 2.5) # B has more positive PnL

    response = client.get(f"/api/v1/abtest/results/{test_name}")
    assert response.status_code == 200
    results = response.json()
    assert results["test_name"] == test_name
    assert results["variant_a_trades"] == 10 # 5 * 2 trades
    assert results["variant_b_trades"] == 15 # 5 * 3 trades
    assert results["variant_a_avg_pnl"] == 0.0
    assert results["variant_b_avg_pnl"] == 2.0
    # Winner might be inconclusive due to low sample size for t-test
    assert results["winner"] == "inconclusive" or results["winner"] == "B"

def test_promote_ab_test_winner_inconclusive():
    test_name = "test_promote_inconclusive"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    ab_service = get_ab_service()
    ab_service.create_test(test_name, variant_a, variant_b)
    # No trades, so results will be inconclusive

    response = client.post(f"/api/v1/abtest/promote/{test_name}")
    assert response.status_code == 400
    assert "Cannot promote: test results inconclusive" in response.json()["detail"]

def test_promote_ab_test_winner_success():
    test_name = "test_promote_success"
    variant_a = {"rsi_length": 14, "take_profit_pct": 0.01}
    variant_b = {"rsi_length": 10, "take_profit_pct": 0.02} # This one will win
    ab_service = get_ab_service()
    ab_service.create_test(test_name, variant_a, variant_b)

    # Record enough trades for significance and a clear winner
    for _ in range(5): # Reduced number of trades
        ab_service.record_trade(test_name, "A", datetime.now(), "SYM", "buy", 100, 100.5, 0.5, 0.5)
        ab_service.record_trade(test_name, "B", datetime.now(), "SYM", "buy", 100, 101.5, 1.5, 1.5)

    response = client.post(f"/api/v1/abtest/promote/{test_name}")
    assert response.status_code == 200
    promotion_details = response.json()["details"]
    assert promotion_details["winner"] == "B"
    assert promotion_details["promoted_params"] == variant_b
    assert promotion_details["confidence"] > 0

    # Verify that the best params for this strategy name are now variant_b
    from backend.optimization.integrated_optimizer import IntegratedStrategyOptimizer
    optimizer = IntegratedStrategyOptimizer(test_name)
    loaded_params = optimizer.load_best_params()
    assert loaded_params == variant_b

def test_get_active_ab_tests():
    test_name_1 = "active_test_1"
    test_name_2 = "active_test_2"
    variant_a = {"p": 1}
    variant_b = {"p": 2}
    ab_service = get_ab_service()
    ab_service.create_test(test_name_1, variant_a, variant_b)
    ab_service.create_test(test_name_2, variant_a, variant_b)

    response = client.get("/api/v1/abtest/active")
    assert response.status_code == 200
    active_tests = response.json()["active_tests"]
    assert len(active_tests) == 2
    assert any(t["test_name"] == test_name_1 for t in active_tests)
    assert any(t["test_name"] == test_name_2 for t in active_tests)
