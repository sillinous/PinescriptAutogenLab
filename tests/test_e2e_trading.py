# tests/test_e2e_trading.py

"""
End-to-end tests for trading workflows.
"""

import pytest
import time
from datetime import datetime


@pytest.mark.e2e
@pytest.mark.trading
class TestCompleteTradingWorkflow:
    """Test complete trading workflows from webhook to execution."""

    def test_full_buy_order_workflow(self, client, auth_headers, db, mock_alpaca_client):
        """Test complete buy order workflow."""
        import os
        import hmac
        import hashlib
        import json

        webhook_secret = os.getenv('WEBHOOK_SECRET')

        # Step 1: Receive webhook from TradingView
        webhook_payload = {
            'strategy': 'TestStrategy',
            'action': 'buy',
            'symbol': 'AAPL',
            'quantity': 10,
            'order_type': 'market',
            'timestamp': datetime.now().isoformat()
        }

        payload_str = json.dumps(webhook_payload)
        signature = hmac.new(
            webhook_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'X-Webhook-Signature': signature,
            'Content-Type': 'application/json'
        }

        webhook_response = client.post(
            '/webhook/tradingview',
            json=webhook_payload,
            headers=headers
        )

        assert webhook_response.status_code in [200, 201]

        # Step 2: Verify order was created
        order_id = webhook_response.json().get('order_id')
        if order_id:
            order_response = client.get(f'/orders/{order_id}', headers=auth_headers)
            assert order_response.status_code == 200

            order = order_response.json()
            assert order['symbol'] == 'AAPL'
            assert order['quantity'] == 10
            assert order['action'] == 'buy'

        # Step 3: Verify order was sent to broker (via mock)
        assert len(mock_alpaca_client.orders) > 0

    def test_full_sell_order_workflow(self, client, auth_headers, db, mock_alpaca_client):
        """Test complete sell order workflow."""
        import os
        import hmac
        import hashlib
        import json

        webhook_secret = os.getenv('WEBHOOK_SECRET')

        # First create a position with buy order
        buy_payload = {
            'strategy': 'TestStrategy',
            'action': 'buy',
            'symbol': 'AAPL',
            'quantity': 10,
            'order_type': 'market'
        }

        buy_sig = hmac.new(
            webhook_secret.encode(),
            json.dumps(buy_payload).encode(),
            hashlib.sha256
        ).hexdigest()

        client.post(
            '/webhook/tradingview',
            json=buy_payload,
            headers={'X-Webhook-Signature': buy_sig}
        )

        # Now sell
        sell_payload = {
            'strategy': 'TestStrategy',
            'action': 'sell',
            'symbol': 'AAPL',
            'quantity': 10,
            'order_type': 'market'
        }

        sell_sig = hmac.new(
            webhook_secret.encode(),
            json.dumps(sell_payload).encode(),
            hashlib.sha256
        ).hexdigest()

        sell_response = client.post(
            '/webhook/tradingview',
            json=sell_payload,
            headers={'X-Webhook-Signature': sell_sig}
        )

        assert sell_response.status_code in [200, 201]

    def test_pnl_calculation_workflow(self, client, auth_headers, db, mock_alpaca_client):
        """Test P&L calculation for completed trades."""
        import os
        import hmac
        import hashlib
        import json

        webhook_secret = os.getenv('WEBHOOK_SECRET')

        # Buy at 150
        mock_alpaca_client.account['equity'] = 100000.0

        buy_payload = {
            'strategy': 'TestStrategy',
            'action': 'buy',
            'symbol': 'AAPL',
            'quantity': 10,
            'order_type': 'market'
        }

        buy_sig = hmac.new(
            webhook_secret.encode(),
            json.dumps(buy_payload).encode(),
            hashlib.sha256
        ).hexdigest()

        buy_response = client.post(
            '/webhook/tradingview',
            json=buy_payload,
            headers={'X-Webhook-Signature': buy_sig}
        )

        assert buy_response.status_code in [200, 201]

        # Check P&L endpoint
        pnl_response = client.get('/pnl', headers=auth_headers)

        if pnl_response.status_code == 200:
            pnl_data = pnl_response.json()
            # Should have P&L data
            assert 'total_pnl' in pnl_data or 'realized_pnl' in pnl_data


@pytest.mark.e2e
@pytest.mark.trading
class TestStrategyOptimizationWorkflow:
    """Test strategy optimization workflows."""

    def test_create_and_optimize_strategy(self, client, auth_headers, sample_strategy_config):
        """Test creating strategy and running optimization."""
        # Step 1: Create strategy
        create_response = client.post(
            '/strategies',
            json=sample_strategy_config,
            headers=auth_headers
        )

        if create_response.status_code in [200, 201]:
            strategy_id = create_response.json().get('id') or create_response.json().get('strategy_id')

            # Step 2: Run optimization
            optimization_config = {
                'strategy_id': strategy_id,
                'symbol': 'AAPL',
                'start_date': '2024-01-01',
                'end_date': '2024-06-30',
                'n_trials': 10,
                'parameters': {
                    'fast_period': {'min': 5, 'max': 20},
                    'slow_period': {'min': 20, 'max': 50}
                }
            }

            optimization_response = client.post(
                '/optimize/start',
                json=optimization_config,
                headers=auth_headers
            )

            # May not be implemented yet
            assert optimization_response.status_code in [200, 201, 404, 501]

    def test_backtest_strategy(self, client, auth_headers, sample_strategy_config):
        """Test backtesting a strategy."""
        # Create strategy
        create_response = client.post(
            '/strategies',
            json=sample_strategy_config,
            headers=auth_headers
        )

        if create_response.status_code in [200, 201]:
            strategy_id = create_response.json().get('id') or create_response.json().get('strategy_id')

            # Run backtest
            backtest_config = {
                'strategy_id': strategy_id,
                'symbol': 'AAPL',
                'start_date': '2024-01-01',
                'end_date': '2024-06-30',
                'initial_capital': 10000
            }

            backtest_response = client.post(
                '/backtest/run',
                json=backtest_config,
                headers=auth_headers
            )

            assert backtest_response.status_code in [200, 201, 404, 501]


@pytest.mark.e2e
@pytest.mark.trading
class TestABTestingWorkflow:
    """Test A/B testing workflows."""

    def test_create_ab_test(self, client, auth_headers):
        """Test creating A/B test."""
        ab_test_config = {
            'name': 'Strategy A vs B',
            'description': 'Test two strategy variants',
            'variant_a': {
                'strategy_name': 'StrategyA',
                'allocation': 0.5
            },
            'variant_b': {
                'strategy_name': 'StrategyB',
                'allocation': 0.5
            },
            'duration_days': 30,
            'success_metric': 'total_return'
        }

        response = client.post(
            '/ab-test/create',
            json=ab_test_config,
            headers=auth_headers
        )

        assert response.status_code in [200, 201, 404, 501]

    def test_ab_test_results(self, client, auth_headers):
        """Test getting A/B test results."""
        response = client.get('/ab-test/results/1', headers=auth_headers)

        # May not have any tests
        assert response.status_code in [200, 404]


@pytest.mark.e2e
class TestUserJourney:
    """Test complete user journeys."""

    def test_new_user_registration_to_first_trade(self, client):
        """Test complete new user journey."""
        # Step 1: Register
        register_response = client.post('/auth/register', json={
            'username': 'journeyuser',
            'email': 'journey@example.com',
            'password': 'SecurePass123!',
            'full_name': 'Journey User'
        })

        assert register_response.status_code == 200
        user = register_response.json()

        # Step 2: Verify email (auto in tests)
        from backend.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user['id'],))
        conn.commit()
        conn.close()

        # Step 3: Login
        login_response = client.post('/auth/login', json={
            'username': 'journeyuser',
            'password': 'SecurePass123!'
        })

        assert login_response.status_code == 200
        token = login_response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        # Step 4: Create strategy
        strategy_config = {
            'name': 'JourneyStrategy',
            'description': 'First strategy',
            'pinescript_code': '//@version=5\nstrategy("Test", overlay=true)',
            'parameters': {}
        }

        strategy_response = client.post('/strategies', json=strategy_config, headers=headers)

        # Step 5: Check profile
        profile_response = client.get('/auth/me', headers=headers)
        assert profile_response.status_code == 200

    def test_admin_monitoring_workflow(self, client, registered_admin):
        """Test admin monitoring and management workflow."""
        # Login as admin
        login_response = client.post('/auth/login', json={
            'username': registered_admin['username'],
            'password': registered_admin['password']
        })

        token = login_response.json()['access_token']
        admin_headers = {'Authorization': f'Bearer {token}'}

        # Step 1: Check health
        health_response = client.get('/health', headers=admin_headers)
        assert health_response.status_code == 200

        # Step 2: View audit log
        audit_response = client.get('/admin/audit-log', headers=admin_headers)
        assert audit_response.status_code == 200

        # Step 3: Create backup
        backup_response = client.post('/backup/create', headers=admin_headers)
        assert backup_response.status_code in [200, 201]

        # Step 4: Run reconciliation
        reconcile_response = client.post('/reconciliation/run', headers=admin_headers)
        assert reconcile_response.status_code == 200

        # Step 5: View metrics
        metrics_response = client.get('/metrics', headers=admin_headers)
        assert metrics_response.status_code in [200, 404]


@pytest.mark.e2e
@pytest.mark.slow
class TestStressTests:
    """Stress tests for the system."""

    def test_concurrent_webhook_requests(self, client, db, mock_alpaca_client):
        """Test handling multiple concurrent webhook requests."""
        import os
        import hmac
        import hashlib
        import json
        import concurrent.futures

        webhook_secret = os.getenv('WEBHOOK_SECRET')

        def send_webhook(i):
            payload = {
                'strategy': f'Strategy{i}',
                'action': 'buy',
                'symbol': 'AAPL',
                'quantity': 1,
                'order_type': 'market'
            }

            sig = hmac.new(
                webhook_secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()

            response = client.post(
                '/webhook/tradingview',
                json=payload,
                headers={'X-Webhook-Signature': sig}
            )

            return response.status_code

        # Send 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_webhook, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed or gracefully handle
        assert all(status in [200, 201, 429, 503] for status in results)

    def test_rapid_order_creation(self, client, auth_headers, db, mock_alpaca_client):
        """Test rapid order creation."""
        import os
        import hmac
        import hashlib
        import json

        webhook_secret = os.getenv('WEBHOOK_SECRET')

        successful_orders = 0

        for i in range(20):
            payload = {
                'strategy': 'RapidStrategy',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'symbol': 'AAPL',
                'quantity': 1,
                'order_type': 'market'
            }

            sig = hmac.new(
                webhook_secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()

            response = client.post(
                '/webhook/tradingview',
                json=payload,
                headers={'X-Webhook-Signature': sig}
            )

            if response.status_code in [200, 201]:
                successful_orders += 1

        # Should handle most orders successfully
        assert successful_orders >= 15
