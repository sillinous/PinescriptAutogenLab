# backend/tests/test_ai_features.py
"""
Tests for AI features including optimization and A/B testing endpoints.
"""

import pytest
import asyncio
from datetime import datetime
import httpx
from backend.app import app
from backend.optimization.integrated_optimizer import _active_optimizations
from backend.ab_testing.ab_service import get_ab_service
from backend.database import get_db
import time


@pytest.fixture
def client():
    """Create test client using httpx ASGITransport."""
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url='http://test') as c:
        yield c


@pytest.fixture
def async_client():
    """Create async test client."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url='http://test')


@pytest.fixture(autouse=True)
def clear_active_optimizations():
    """Fixture to clear active optimizations before each test."""
    _active_optimizations.clear()
    # Clear AB test data as well
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ab_tests")
    cursor.execute("DELETE FROM ab_test_trades")
    conn.commit()
    conn.close()
    yield


@pytest.mark.asyncio
async def test_get_available_strategies(async_client):
    async with async_client as client:
        response = await client.get("/api/v1/ai/optimize/strategies")
        assert response.status_code == 200
        strategies = response.json()
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert any(s["type"] == "rsi" for s in strategies)


@pytest.mark.asyncio
async def test_start_strategy_optimization_success(async_client):
    strategy_name = "test_rsi_opt"
    async with async_client as client:
        response = await client.post(
            "/api/v1/ai/optimize/start",
            json={"strategy_name": strategy_name, "strategy_type": "rsi", "n_trials": 1}
        )
        assert response.status_code == 200
        assert response.json()["message"] == f"Optimization for strategy '{strategy_name}' started successfully."


@pytest.mark.asyncio
async def test_start_strategy_optimization_invalid_type(async_client):
    async with async_client as client:
        response = await client.post(
            "/api/v1/ai/optimize/start",
            json={"strategy_name": "invalid_opt", "strategy_type": "non_existent", "n_trials": 1}
        )
        assert response.status_code == 400
        assert "Unknown strategy type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_optimization_status_not_found(async_client):
    async with async_client as client:
        response = await client.get("/api/v1/ai/optimize/status/non_existent_opt")
        assert response.status_code == 404
        assert "not found or not started" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_ab_test_success(async_client):
    test_name = "test_ab_rsi"
    variant_a = {"rsi_length": 14, "rsi_oversold": 30, "rsi_overbought": 70}
    variant_b = {"rsi_length": 10, "rsi_oversold": 20, "rsi_overbought": 80}
    async with async_client as client:
        response = await client.post(
            "/api/v1/ai/abtest/create",
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
        active_tests_response = await client.get("/api/v1/ai/abtest/active")
        assert active_tests_response.status_code == 200
        active_tests = active_tests_response.json()["active_tests"]
        assert any(t["test_name"] == test_name for t in active_tests)


@pytest.mark.asyncio
async def test_create_ab_test_duplicate_name(async_client):
    test_name = "duplicate_test"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    async with async_client as client:
        await client.post(
            "/api/v1/ai/abtest/create",
            json={"test_name": test_name, "variant_a_params": variant_a, "variant_b_params": variant_b}
        )
        response = await client.post(
            "/api/v1/ai/abtest/create",
            json={"test_name": test_name, "variant_a_params": variant_a, "variant_b_params": variant_b}
        )
        assert response.status_code == 400
        assert "UNIQUE constraint failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_ab_test_results_no_trades(async_client):
    test_name = "test_ab_no_trades"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    async with async_client as client:
        await client.post(
            "/api/v1/ai/abtest/create",
            json={"test_name": test_name, "variant_a_params": variant_a, "variant_b_params": variant_b}
        )
        response = await client.get(f"/api/v1/ai/abtest/results/{test_name}")
        assert response.status_code == 200
        results = response.json()
        assert results["test_name"] == test_name
        assert results["variant_a_trades"] == 0
        assert results["variant_b_trades"] == 0
        assert results["winner"] == "inconclusive"


@pytest.mark.asyncio
async def test_get_ab_test_results_with_trades(async_client):
    test_name = "test_ab_with_trades"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    ab_service = get_ab_service()
    ab_service.create_test(test_name, variant_a, variant_b)

    # Record some trades
    for _ in range(5):
        ab_service.record_trade(test_name, "A", datetime.now(), "BTCUSDT", "buy", 100, 101, 1.0, 1.0)
        ab_service.record_trade(test_name, "A", datetime.now(), "BTCUSDT", "sell", 100, 99, -1.0, -1.0)
        ab_service.record_trade(test_name, "B", datetime.now(), "ETHUSDT", "buy", 200, 203, 3.0, 1.5)
        ab_service.record_trade(test_name, "B", datetime.now(), "ETHUSDT", "sell", 200, 198, -2.0, -1.0)
        ab_service.record_trade(test_name, "B", datetime.now(), "ETHUSDT", "buy", 200, 205, 5.0, 2.5)

    async with async_client as client:
        response = await client.get(f"/api/v1/ai/abtest/results/{test_name}")
        assert response.status_code == 200
        results = response.json()
        assert results["test_name"] == test_name
        assert results["variant_a_trades"] == 10  # 5 * 2 trades
        assert results["variant_b_trades"] == 15  # 5 * 3 trades


@pytest.mark.asyncio
async def test_promote_ab_test_winner_inconclusive(async_client):
    test_name = "test_promote_inconclusive"
    variant_a = {"rsi_length": 14}
    variant_b = {"rsi_length": 10}
    ab_service = get_ab_service()
    ab_service.create_test(test_name, variant_a, variant_b)

    async with async_client as client:
        response = await client.post(f"/api/v1/ai/abtest/promote/{test_name}")
        assert response.status_code == 400
        assert "Cannot promote: test results inconclusive" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_active_ab_tests(async_client):
    test_name_1 = "active_test_1"
    test_name_2 = "active_test_2"
    variant_a = {"p": 1}
    variant_b = {"p": 2}
    ab_service = get_ab_service()
    ab_service.create_test(test_name_1, variant_a, variant_b)
    ab_service.create_test(test_name_2, variant_a, variant_b)

    async with async_client as client:
        response = await client.get("/api/v1/ai/abtest/active")
        assert response.status_code == 200
        active_tests = response.json()["active_tests"]
        assert len(active_tests) >= 2
        assert any(t["test_name"] == test_name_1 for t in active_tests)
        assert any(t["test_name"] == test_name_2 for t in active_tests)
