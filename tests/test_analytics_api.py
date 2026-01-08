# tests/test_analytics_api.py
"""
Tests for the Analytics API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestTradeHistoryEndpoints:
    """Tests for trade history analytics."""

    def test_get_trade_history_default(self, client):
        """Test getting trade history with default parameters."""
        response = client.get('/api/v1/analytics/trades/history')
        assert response.status_code == 200
        data = response.json()
        assert 'trades' in data
        assert 'total' in data
        assert 'limit' in data
        assert 'offset' in data

    def test_get_trade_history_with_filters(self, client):
        """Test getting trade history with filters."""
        response = client.get('/api/v1/analytics/trades/history?limit=10&symbol=BTC_USDT&status=filled')
        assert response.status_code == 200
        data = response.json()
        assert data['limit'] == 10

    def test_get_trade_metrics_all(self, client):
        """Test getting trade metrics for all time."""
        response = client.get('/api/v1/analytics/trades/metrics?period=all')
        assert response.status_code == 200
        data = response.json()
        assert 'total_trades' in data
        assert 'win_rate' in data
        assert 'realized_pnl' in data

    def test_get_trade_metrics_month(self, client):
        """Test getting trade metrics for past month."""
        response = client.get('/api/v1/analytics/trades/metrics?period=month')
        assert response.status_code == 200

    def test_get_daily_performance(self, client):
        """Test getting daily performance breakdown."""
        response = client.get('/api/v1/analytics/trades/daily-performance?days=30')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_trades_by_symbol(self, client):
        """Test getting trades grouped by symbol."""
        response = client.get('/api/v1/analytics/trades/by-symbol?period=month')
        assert response.status_code == 200
        data = response.json()
        assert 'symbols' in data
        assert 'total_symbols' in data


class TestPortfolioEndpoints:
    """Tests for portfolio monitoring."""

    def test_get_portfolio_positions(self, client):
        """Test getting current positions."""
        response = client.get('/api/v1/analytics/portfolio/positions')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_portfolio_summary(self, client):
        """Test getting portfolio summary."""
        response = client.get('/api/v1/analytics/portfolio/summary')
        assert response.status_code == 200
        data = response.json()
        assert 'total_equity' in data
        assert 'total_cash' in data
        assert 'position_count' in data

    def test_get_equity_curve(self, client):
        """Test getting equity curve data."""
        response = client.get('/api/v1/analytics/portfolio/equity-curve?days=30')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestRiskEndpoints:
    """Tests for risk metrics."""

    def test_get_risk_metrics(self, client):
        """Test getting risk metrics."""
        response = client.get('/api/v1/analytics/risk/metrics')
        assert response.status_code == 200
        data = response.json()
        assert 'position_count' in data
        assert 'total_exposure' in data

    def test_get_risk_exposure(self, client):
        """Test getting risk exposure breakdown."""
        response = client.get('/api/v1/analytics/risk/exposure')
        assert response.status_code == 200
        data = response.json()
        assert 'long_exposure' in data
        assert 'short_exposure' in data
        assert 'net_exposure' in data


class TestStrategyEndpoints:
    """Tests for strategy performance."""

    def test_list_strategies(self, client):
        """Test listing all strategies."""
        response = client.get('/api/v1/analytics/strategies/list')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_strategy_performance(self, client):
        """Test getting strategy performance."""
        response = client.get('/api/v1/analytics/strategies/TestStrategy/performance')
        assert response.status_code == 200
        data = response.json()
        assert 'strategy_name' in data
        assert 'total_trades' in data
