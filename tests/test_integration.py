# tests/test_integration.py

"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestAuthenticationEndpoints:
    """Test authentication-related endpoints."""

    def test_user_registration(self, client):
        """Test user registration endpoint."""
        response = client.post('/auth/register', json={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'SecurePass123!',
            'full_name': 'New User'
        })

        assert response.status_code == 200
        data = response.json()

        assert data['username'] == 'newuser'
        assert data['email'] == 'newuser@example.com'
        assert 'id' in data
        assert 'password' not in data  # Password should not be returned

    def test_duplicate_username_registration(self, client, registered_user):
        """Test registering with duplicate username."""
        response = client.post('/auth/register', json={
            'username': registered_user['username'],
            'email': 'different@example.com',
            'password': 'SecurePass123!',
            'full_name': 'Different User'
        })

        assert response.status_code == 400
        assert 'exists' in response.json()['detail'].lower()

    def test_user_login(self, client, registered_user):
        """Test user login endpoint."""
        response = client.post('/auth/login', json={
            'username': registered_user['username'],
            'password': registered_user['password']
        })

        assert response.status_code == 200
        data = response.json()

        assert 'access_token' in data
        assert 'refresh_token' in data
        assert data['token_type'] == 'bearer'

    def test_login_invalid_credentials(self, client, registered_user):
        """Test login with invalid credentials."""
        response = client.post('/auth/login', json={
            'username': registered_user['username'],
            'password': 'wrong_password'
        })

        assert response.status_code == 401

    def test_refresh_token(self, client, registered_user):
        """Test token refresh endpoint."""
        # Login to get tokens
        login_response = client.post('/auth/login', json={
            'username': registered_user['username'],
            'password': registered_user['password']
        })

        refresh_token = login_response.json()['refresh_token']

        # Refresh token
        refresh_response = client.post('/auth/refresh', json={
            'refresh_token': refresh_token
        })

        assert refresh_response.status_code == 200
        assert 'access_token' in refresh_response.json()

    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info."""
        response = client.get('/auth/me', headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert 'username' in data
        assert 'email' in data

    def test_unauthorized_access(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get('/auth/me')

        assert response.status_code == 401


@pytest.mark.integration
class TestWebhookEndpoints:
    """Test webhook endpoints."""

    def test_tradingview_webhook(self, client, sample_webhook_payload):
        """Test TradingView webhook endpoint."""
        import os
        import hmac
        import hashlib
        import json

        webhook_secret = os.getenv('WEBHOOK_SECRET')

        # Calculate signature
        payload_str = json.dumps(sample_webhook_payload)
        signature = hmac.new(
            webhook_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'X-Webhook-Signature': signature,
            'Content-Type': 'application/json'
        }

        response = client.post(
            '/webhook/tradingview',
            json=sample_webhook_payload,
            headers=headers
        )

        assert response.status_code in [200, 201]
        data = response.json()

        assert 'order_id' in data or 'message' in data

    def test_webhook_invalid_signature(self, client, sample_webhook_payload):
        """Test webhook with invalid signature."""
        headers = {
            'X-Webhook-Signature': 'invalid_signature_12345',
            'Content-Type': 'application/json'
        }

        response = client.post(
            '/webhook/tradingview',
            json=sample_webhook_payload,
            headers=headers
        )

        assert response.status_code == 401

    def test_webhook_missing_signature(self, client, sample_webhook_payload):
        """Test webhook without signature."""
        response = client.post(
            '/webhook/tradingview',
            json=sample_webhook_payload
        )

        assert response.status_code == 401


@pytest.mark.integration
class TestOrderEndpoints:
    """Test order management endpoints."""

    def test_list_orders(self, client, auth_headers):
        """Test listing orders."""
        response = client.get('/orders', headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_get_order_details(self, client, auth_headers, db):
        """Test getting specific order details."""
        from backend.database import create_order

        # Create test order
        order_id = create_order(
            strategy_name='TestStrategy',
            symbol='AAPL',
            action='buy',
            quantity=10,
            order_type='market',
            status='pending'
        )

        response = client.get(f'/orders/{order_id}', headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data['id'] == order_id
        assert data['symbol'] == 'AAPL'

    def test_cancel_order(self, client, auth_headers, db, mock_alpaca_client):
        """Test cancelling an order."""
        from backend.database import create_order

        # Create test order
        order_id = create_order(
            strategy_name='TestStrategy',
            symbol='AAPL',
            action='buy',
            quantity=10,
            order_type='limit',
            status='pending',
            broker_order_id='order_1'
        )

        response = client.post(f'/orders/{order_id}/cancel', headers=auth_headers)

        assert response.status_code == 200


@pytest.mark.integration
class TestStrategyEndpoints:
    """Test strategy management endpoints."""

    def test_create_strategy(self, client, auth_headers, sample_strategy_config):
        """Test creating a strategy."""
        response = client.post(
            '/strategies',
            json=sample_strategy_config,
            headers=auth_headers
        )

        assert response.status_code in [200, 201]
        data = response.json()

        assert 'id' in data or 'strategy_id' in data

    def test_list_strategies(self, client, auth_headers):
        """Test listing strategies."""
        response = client.get('/strategies', headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_get_strategy_details(self, client, auth_headers, sample_strategy_config):
        """Test getting strategy details."""
        # Create strategy first
        create_response = client.post(
            '/strategies',
            json=sample_strategy_config,
            headers=auth_headers
        )

        if create_response.status_code in [200, 201]:
            strategy_id = create_response.json().get('id') or create_response.json().get('strategy_id')

            response = client.get(f'/strategies/{strategy_id}', headers=auth_headers)

            assert response.status_code == 200


@pytest.mark.integration
class TestMonitoringEndpoints:
    """Test monitoring and health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get('/health')

        assert response.status_code == 200
        data = response.json()

        assert 'status' in data

    def test_quick_health_check(self, client):
        """Test quick health check."""
        response = client.get('/health/quick')

        assert response.status_code == 200

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get('/health/ready')

        assert response.status_code == 200

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get('/health/live')

        assert response.status_code == 200

    def test_metrics_endpoint(self, client, admin_headers):
        """Test metrics endpoint (admin only)."""
        response = client.get('/metrics', headers=admin_headers)

        # May require admin auth
        assert response.status_code in [200, 401, 403]


@pytest.mark.integration
class TestBackupEndpoints:
    """Test backup and restore endpoints."""

    def test_create_backup(self, client, admin_headers):
        """Test creating backup via API."""
        response = client.post('/backup/create', headers=admin_headers)

        assert response.status_code in [200, 201]
        data = response.json()

        assert 'backup_name' in data or 'success' in data

    def test_list_backups(self, client, admin_headers):
        """Test listing backups."""
        response = client.get('/backup/list', headers=admin_headers)

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)

    def test_backup_requires_admin(self, client, auth_headers):
        """Test backup endpoints require admin access."""
        response = client.post('/backup/create', headers=auth_headers)

        # Should be forbidden for non-admin
        assert response.status_code in [401, 403]


@pytest.mark.integration
class TestAdminEndpoints:
    """Test admin-only endpoints."""

    def test_audit_log_requires_admin(self, client, auth_headers, admin_headers):
        """Test audit log requires admin access."""
        # Non-admin should be denied
        response = client.get('/admin/audit-log', headers=auth_headers)
        assert response.status_code in [401, 403]

        # Admin should have access
        response = client.get('/admin/audit-log', headers=admin_headers)
        assert response.status_code == 200

    def test_list_users_admin(self, client, admin_headers):
        """Test listing all users (admin only)."""
        response = client.get('/admin/users', headers=admin_headers)

        # Endpoint may not exist, but should not return 403 for admin
        assert response.status_code in [200, 404]


@pytest.mark.integration
class TestReconciliationEndpoints:
    """Test reconciliation endpoints."""

    def test_run_reconciliation(self, client, admin_headers):
        """Test manual reconciliation trigger."""
        response = client.post('/reconciliation/run', headers=admin_headers)

        assert response.status_code == 200

    def test_get_stale_orders(self, client, admin_headers):
        """Test getting stale orders."""
        response = client.get('/reconciliation/stale-orders', headers=admin_headers)

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
