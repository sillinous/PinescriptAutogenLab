# tests/conftest.py

"""
Pytest fixtures and configuration for test suite.
"""

import pytest
import os
import sys
import tempfile
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Generator, Dict, Any
import asyncio
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test environment BEFORE importing backend modules
os.environ['TESTING'] = 'true'
os.environ['WEBHOOK_SECRET'] = 'test_webhook_secret_key_12345'
os.environ['JWT_SECRET'] = 'test_jwt_secret_key_67890'
os.environ['ENCRYPTION_KEY'] = 'test_encryption_key_abcdefghijklmnopqrstuvwxyz123456'
os.environ['SMTP_ENABLED'] = 'false'

# Check for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_torch: skip if PyTorch not installed")
    config.addinivalue_line("markers", "requires_pyotp: skip if pyotp not installed")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    skip_torch = pytest.mark.skip(reason="PyTorch not installed")
    skip_pyotp = pytest.mark.skip(reason="pyotp not installed")

    for item in items:
        # Auto-skip deep learning tests if torch not available
        if "deep_learning" in item.nodeid or "TestLSTM" in item.nodeid or "TestTransformer" in item.nodeid or "TestEnsemble" in item.nodeid:
            if not TORCH_AVAILABLE:
                item.add_marker(skip_torch)

        # Skip 2FA tests if pyotp not available
        if "two_factor" in item.nodeid or "2fa" in item.nodeid.lower():
            if not PYOTP_AVAILABLE:
                item.add_marker(skip_pyotp)

        # Check for explicit markers
        if item.get_closest_marker("requires_torch") and not TORCH_AVAILABLE:
            item.add_marker(skip_torch)
        if item.get_closest_marker("requires_pyotp") and not PYOTP_AVAILABLE:
            item.add_marker(skip_pyotp)


@pytest.fixture(scope="function")
def isolated_db_path() -> Generator[str, None, None]:
    """Create an isolated temporary database for each test."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, f'test_{uuid.uuid4().hex[:8]}.db')

    yield db_path

    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(scope="function")
def db(isolated_db_path: str, monkeypatch) -> Generator:
    """Create a fresh isolated database for each test."""
    from pathlib import Path

    # Create a new Path object for the test database
    test_db_path = Path(isolated_db_path)

    # Patch the DB_PATH before importing/initializing
    import backend.database as db_module
    original_db_path = db_module.DB_PATH

    monkeypatch.setattr(db_module, 'DB_PATH', test_db_path)

    # Override get_db to use check_same_thread=False for test isolation
    original_get_db = db_module.get_db
    def test_get_db():
        conn = sqlite3.connect(str(test_db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    monkeypatch.setattr(db_module, 'get_db', test_get_db)

    # Initialize the test database
    db_module.init_db()

    yield test_db_path

    # Cleanup handled by isolated_db_path fixture


@pytest.fixture(scope="function")
def client(db) -> TestClient:
    """Create a test client for the FastAPI app."""
    from backend.app import app

    return TestClient(app)


@pytest.fixture
def test_user_data() -> Dict[str, str]:
    """Sample user registration data."""
    return {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'TestPassword123!',
        'full_name': 'Test User'
    }


@pytest.fixture
def test_admin_data() -> Dict[str, str]:
    """Sample admin user data."""
    return {
        'username': 'admin',
        'email': 'admin@example.com',
        'password': 'AdminPassword123!',
        'full_name': 'Admin User'
    }


@pytest.fixture
def registered_user(client: TestClient, test_user_data: Dict[str, str]) -> Dict[str, Any]:
    """Create and return a registered user."""
    response = client.post('/auth/register', json=test_user_data)
    assert response.status_code == 200
    user = response.json()

    # Verify email automatically in tests
    from backend.database import get_db
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user['id'],))
    conn.commit()
    conn.close()

    return {**user, 'password': test_user_data['password']}


@pytest.fixture
def registered_admin(client: TestClient, test_admin_data: Dict[str, str]) -> Dict[str, Any]:
    """Create and return an admin user."""
    response = client.post('/auth/register', json=test_admin_data)
    assert response.status_code == 200
    user = response.json()

    # Make user admin and verify email
    from backend.database import get_db
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_admin = 1, email_verified = 1 WHERE id = ?", (user['id'],))
    conn.commit()
    conn.close()

    return {**user, 'password': test_admin_data['password'], 'is_admin': True}


@pytest.fixture
def user_token(client: TestClient, registered_user: Dict[str, Any]) -> str:
    """Get JWT token for registered user."""
    response = client.post('/auth/login', json={
        'username': registered_user['username'],
        'password': registered_user['password']
    })
    assert response.status_code == 200
    return response.json()['access_token']


@pytest.fixture
def admin_token(client: TestClient, registered_admin: Dict[str, Any]) -> str:
    """Get JWT token for admin user."""
    response = client.post('/auth/login', json={
        'username': registered_admin['username'],
        'password': registered_admin['password']
    })
    assert response.status_code == 200
    return response.json()['access_token']


@pytest.fixture
def auth_headers(user_token: str) -> Dict[str, str]:
    """Authorization headers with user token."""
    return {'Authorization': f'Bearer {user_token}'}


@pytest.fixture
def admin_headers(admin_token: str) -> Dict[str, str]:
    """Authorization headers with admin token."""
    return {'Authorization': f'Bearer {admin_token}'}


@pytest.fixture
def sample_webhook_payload() -> Dict[str, Any]:
    """Sample TradingView webhook payload."""
    return {
        'strategy': 'TestStrategy',
        'action': 'buy',
        'symbol': 'AAPL',
        'quantity': 10,
        'order_type': 'market',
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_strategy_config() -> Dict[str, Any]:
    """Sample strategy configuration."""
    return {
        'name': 'TestStrategy',
        'description': 'Test strategy for unit tests',
        'pinescript_code': '''
//@version=5
strategy("Test Strategy", overlay=true)
longCondition = ta.crossover(ta.sma(close, 14), ta.sma(close, 28))
if (longCondition)
    strategy.entry("Long", strategy.long)
        ''',
        'parameters': {
            'fast_period': 14,
            'slow_period': 28
        },
        'risk_params': {
            'max_position_size': 100,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 5.0
        }
    }


@pytest.fixture
def mock_alpaca_client(monkeypatch):
    """Mock Alpaca API client."""
    class MockAlpacaClient:
        def __init__(self, *args, **kwargs):
            self.orders = {}
            self.positions = {}
            self.account = {
                'id': 'test_account_id',
                'equity': 100000.0,
                'cash': 50000.0,
                'buying_power': 200000.0
            }

        async def get_account(self):
            return self.account

        async def submit_order(self, symbol: str, qty: int, side: str, type: str, **kwargs):
            order_id = f'order_{len(self.orders) + 1}'
            order = {
                'id': order_id,
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': type,
                'status': 'filled',
                'filled_qty': qty,
                'filled_avg_price': 150.0,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            self.orders[order_id] = order
            return order

        async def get_order(self, order_id: str):
            return self.orders.get(order_id)

        async def cancel_order(self, order_id: str):
            if order_id in self.orders:
                self.orders[order_id]['status'] = 'canceled'
            return self.orders.get(order_id)

        async def get_position(self, symbol: str):
            return self.positions.get(symbol)

        async def list_positions(self):
            return list(self.positions.values())

    mock = MockAlpacaClient()

    # Patch the alpaca client getter
    def mock_get_client(*args, **kwargs):
        return mock

    from backend.brokers import alpaca_client as alpaca_integration
    monkeypatch.setattr(alpaca_integration, 'get_alpaca_client', mock_get_client)

    return mock


@pytest.fixture
def encryption_key() -> str:
    """Test encryption key."""
    return 'test_encryption_key_abcdefghijklmnopqrstuvwxyz123456'


@pytest.fixture
def cleanup_temp_files():
    """Cleanup temporary files after tests."""
    temp_files = []

    yield temp_files

    # Cleanup
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)


# Event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Helper functions for tests
def create_test_order(client: TestClient, headers: Dict[str, str], symbol: str = "AAPL",
                      action: str = "buy", quantity: int = 10) -> Dict[str, Any]:
    """Helper to create a test order."""
    payload = {
        'strategy': 'TestStrategy',
        'action': action,
        'symbol': symbol,
        'quantity': quantity,
        'order_type': 'market'
    }
    response = client.post('/webhook/tradingview', json=payload, headers=headers)
    return response.json()


def verify_order_status(client: TestClient, headers: Dict[str, str],
                        order_id: int, expected_status: str) -> bool:
    """Helper to verify order status."""
    response = client.get(f'/orders/{order_id}', headers=headers)
    if response.status_code != 200:
        return False
    order = response.json()
    return order.get('status') == expected_status
