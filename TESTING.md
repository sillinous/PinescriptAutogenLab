# Testing Guide for PinescriptAutogenLab

Comprehensive testing documentation for the platform.

## Table of Contents

1. [Test Suite Overview](#test-suite-overview)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Writing Tests](#writing-tests)
5. [Coverage Requirements](#coverage-requirements)
6. [CI/CD Integration](#cicd-integration)

---

## Test Suite Overview

### Test Statistics

- **Total Test Files**: 7
- **Test Categories**: Unit, Integration, E2E, Security, WebSocket, Trading
- **Coverage Target**: 80%
- **Test Framework**: pytest
- **Async Support**: pytest-asyncio

### Test Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── test_encryption.py              # Encryption service tests
├── test_two_factor.py              # 2FA tests
├── test_email_verification.py      # Email verification tests
├── test_password_reset.py          # Password reset tests
├── test_reliability.py             # Retry, reconciliation, backups
├── test_integration.py             # API endpoint integration tests
├── test_websocket.py               # WebSocket tests
└── test_e2e_trading.py             # End-to-end trading workflows
```

---

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Or install from requirements
pip install -r requirements.txt
```

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_encryption.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_encryption"
```

### Using Test Script

```bash
# Make script executable
chmod +x scripts/test.sh

# Run all tests
./scripts/test.sh

# Run specific category
./scripts/test.sh unit
./scripts/test.sh integration
./scripts/test.sh e2e
./scripts/test.sh security
./scripts/test.sh websocket
./scripts/test.sh trading

# Run fast tests only (exclude slow tests)
./scripts/test.sh fast
```

### Docker Testing

```bash
# Run tests in Docker
docker-compose exec backend pytest

# Run with coverage
docker-compose exec backend pytest --cov=backend --cov-report=term

# Run specific test
docker-compose exec backend pytest tests/test_encryption.py -v
```

---

## Test Categories

### Unit Tests

**Marker**: `@pytest.mark.unit`

**Purpose**: Test individual components in isolation

**Files**:
- `test_encryption.py` - Encryption service
- `test_two_factor.py` - 2FA functionality
- `test_email_verification.py` - Email verification
- `test_password_reset.py` - Password reset
- `test_reliability.py` - Retry, reconciliation, backups

**Run**:
```bash
pytest -m unit
```

**Examples**:
- Encryption/decryption roundtrip
- 2FA secret generation
- Password hashing
- Token generation
- Backup creation

### Integration Tests

**Marker**: `@pytest.mark.integration`

**Purpose**: Test component interactions and API endpoints

**Files**:
- `test_integration.py` - API endpoint tests

**Run**:
```bash
pytest -m integration
```

**Examples**:
- User registration flow
- Login and token refresh
- Order creation via API
- Webhook processing
- Admin endpoints

### End-to-End Tests

**Marker**: `@pytest.mark.e2e`

**Purpose**: Test complete user workflows

**Files**:
- `test_e2e_trading.py` - Complete trading workflows

**Run**:
```bash
pytest -m e2e
```

**Examples**:
- Webhook → Order → Execution flow
- Strategy creation → Optimization
- User registration → First trade
- Admin monitoring workflow

### Security Tests

**Marker**: `@pytest.mark.security`

**Purpose**: Test security features

**Files**:
- `test_encryption.py`
- `test_two_factor.py`
- `test_email_verification.py`
- `test_password_reset.py`

**Run**:
```bash
pytest -m security
```

**Examples**:
- Password strength validation
- Token expiration
- Signature verification
- Encryption key rotation
- 2FA bypass attempts

### WebSocket Tests

**Marker**: `@pytest.mark.websocket`

**Purpose**: Test real-time WebSocket functionality

**Files**:
- `test_websocket.py`

**Run**:
```bash
pytest -m websocket
```

**Examples**:
- Connection establishment
- Message broadcasting
- User-specific messages
- Connection cleanup

### Trading Tests

**Marker**: `@pytest.mark.trading`

**Purpose**: Test trading-specific functionality

**Files**:
- `test_e2e_trading.py`

**Run**:
```bash
pytest -m trading
```

**Examples**:
- Order execution
- P&L calculation
- Position tracking
- Strategy optimization

### Slow Tests

**Marker**: `@pytest.mark.slow`

**Purpose**: Mark long-running tests

**Run all except slow**:
```bash
pytest -m "not slow"
```

---

## Writing Tests

### Test Structure

```python
import pytest
from backend.service import MyService

@pytest.mark.unit
class TestMyService:
    """Test MyService functionality."""

    def test_basic_operation(self, db):
        """Test basic operation."""
        service = MyService()
        result = service.do_something()
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_operation(self, db):
        """Test async operation."""
        service = MyService()
        result = await service.do_something_async()
        assert result is not None
```

### Using Fixtures

#### Built-in Fixtures

```python
def test_with_database(db):
    """Test using database fixture."""
    # db fixture provides fresh database for each test
    pass

def test_with_client(client):
    """Test using test client."""
    response = client.get('/health')
    assert response.status_code == 200

def test_with_auth(client, auth_headers):
    """Test with authentication."""
    response = client.get('/auth/me', headers=auth_headers)
    assert response.status_code == 200

def test_with_admin(client, admin_headers):
    """Test admin endpoint."""
    response = client.get('/admin/users', headers=admin_headers)
    assert response.status_code == 200
```

#### Available Fixtures

From `conftest.py`:
- `db` - Fresh database for each test
- `client` - FastAPI TestClient
- `test_user_data` - Sample user data
- `test_admin_data` - Sample admin data
- `registered_user` - Pre-registered user
- `registered_admin` - Pre-registered admin
- `user_token` - JWT token for user
- `admin_token` - JWT token for admin
- `auth_headers` - Authorization headers (user)
- `admin_headers` - Authorization headers (admin)
- `sample_webhook_payload` - TradingView webhook data
- `sample_strategy_config` - Strategy configuration
- `mock_alpaca_client` - Mocked broker API
- `encryption_key` - Test encryption key
- `cleanup_temp_files` - Cleanup helper

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await some_async_function()
    assert result is not None
```

### Testing Exceptions

```python
def test_exception_handling():
    """Test exception is raised."""
    service = MyService()

    with pytest.raises(ValueError):
        service.invalid_operation()

    with pytest.raises(ValueError, match="Invalid input"):
        service.invalid_operation()
```

### Testing API Endpoints

```python
def test_api_endpoint(client, auth_headers):
    """Test API endpoint."""
    # POST request
    response = client.post('/api/resource', json={
        'name': 'test',
        'value': 123
    }, headers=auth_headers)

    assert response.status_code == 201
    data = response.json()
    assert data['name'] == 'test'

    # GET request
    response = client.get('/api/resource/1', headers=auth_headers)
    assert response.status_code == 200
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("password123", False),
    ("Password123!", True),
    ("short", False),
    ("VerySecurePassword123!", True),
])
def test_password_strength(input, expected):
    """Test password strength validation."""
    result = validate_password_strength(input)
    assert result == expected
```

---

## Coverage Requirements

### Target Coverage

- **Overall**: 80% minimum
- **Critical modules**: 90%+
  - Security (encryption, auth)
  - Trading (order execution)
  - Reliability (backups, reconciliation)

### Viewing Coverage

```bash
# Generate HTML coverage report
pytest --cov=backend --cov-report=html

# Open in browser
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Configuration

In `pytest.ini`:
```ini
[coverage:run]
source = backend
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio

    - name: Run tests
      run: |
        pytest --cov=backend --cov-report=xml --cov-report=term

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

---

## Best Practices

### 1. Test Isolation

- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Clear Test Names

```python
# Good
def test_user_registration_with_valid_data():
    pass

# Bad
def test_1():
    pass
```

### 3. One Assertion Per Test

```python
# Good
def test_user_has_email():
    user = create_user()
    assert user.email is not None

def test_user_has_username():
    user = create_user()
    assert user.username is not None

# Acceptable for related assertions
def test_user_creation():
    user = create_user()
    assert user.id is not None
    assert user.created_at is not None
```

### 4. Use Descriptive Assertions

```python
# Good
assert response.status_code == 200, f"Expected 200, got {response.status_code}"

# Better
assert response.status_code == 200, \
    f"Health check failed: {response.json()}"
```

### 5. Test Edge Cases

- Empty inputs
- Null values
- Boundary conditions
- Error conditions

### 6. Mock External Dependencies

```python
def test_api_call(monkeypatch):
    """Test with mocked external API."""
    def mock_api_call(*args, **kwargs):
        return {'status': 'success'}

    monkeypatch.setattr('module.api_call', mock_api_call)

    result = function_that_calls_api()
    assert result['status'] == 'success'
```

---

## Troubleshooting Tests

### Import Errors

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Database Locked Errors

```python
# Use in-memory database for tests
import os
os.environ['DATABASE_PATH'] = ':memory:'
```

### Async Test Failures

```python
# Ensure asyncio mode is set
# In pytest.ini:
[pytest]
asyncio_mode = auto
```

### Fixture Not Found

```python
# Ensure conftest.py is in correct location
# Check fixture scope matches usage
```

---

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
