# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PinescriptAutogenLab (PineLab) is an AI-powered trading platform that combines cryptocurrency market data from Crypto.com with advanced machine learning capabilities including deep learning (LSTM, Transformers), reinforcement learning, and signal aggregation. The platform features a FastAPI backend with SQLite/PostgreSQL database and a React + Vite frontend with Tailwind CSS.

## Commands

### Development

**Backend:**
```bash
# Start backend server (from project root)
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Alternative from backend directory
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn app:app --reload --port 8080
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev  # Starts on http://localhost:5173
npm run build  # Production build
npm run preview  # Preview production build
```

### Testing

```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m trading
pytest -m websocket

# Run specific test file
pytest tests/test_encryption.py

# Run with verbose output and coverage report
pytest -v --cov=backend --cov-report=html --cov-report=term

# Using test script
chmod +x scripts/test.sh
./scripts/test.sh
```

### Docker

```bash
# Development mode
make dev-up
make dev-down
make dev-logs

# Production mode
make prod-up
make prod-down
make prod-logs

# Alternative docker-compose commands
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.prod.yml --profile production up -d
```

### Kubernetes

```bash
# Deploy all resources
kubectl apply -f k8s/

# Check status
kubectl get all -n pinelab
kubectl get pods -n pinelab

# View logs
kubectl logs -f deployment/pinelab-backend -n pinelab
```

## Architecture

### Backend Structure

The backend follows a modular architecture with clear separation of concerns:

**Core Application (`backend/app.py`):**
- FastAPI application with CORS middleware
- Integrates AI Trading router (`api_ai.py`) and Deep Learning router (`api_deep_learning.py`)
- Crypto.com API integration for market data
- SQLite caching with `cache.db` for candle/price data
- Symbol normalization for crypto pairs (e.g., BTC_USDT, ETH_USD)

**AI/ML Modules (`backend/ai/`):**
- `deep_learning/`: LSTM predictor, Transformer predictor, ensemble models
- `reinforcement_learning/`: Trading agent and environment
- `features/`: Feature engineering for time series
- `signal_aggregator.py`: Aggregates signals from multiple sources
- Meta-learning and evolution modules for strategy optimization

**Integrations (`backend/integrations/`):**
- `tradingview/`: Webhook handler, chart service for TradingView signals
- TradingView webhook processing with signal enrichment

**Infrastructure Modules:**
- `auth/`: JWT authentication, 2FA, email verification, password reset
- `brokers/`: Alpaca integration for live/paper trading
- `monitoring/`: Health checks, metrics, audit logging
- `reliability/`: Backup service, reconciliation, retry handler with exponential backoff
- `security/`: Encryption service (Fernet), webhook signature verification
- `notifications/`: Email service (SMTP)
- `websocket/`: Real-time updates service
- `middleware/`: Centralized error handling
- `ab_testing/`: A/B testing service for experiments
- `optimization/`: Optuna integration, backtester, integrated optimizer
- `order_service.py`: Trade execution service
- `shutdown_handler.py`: Graceful shutdown handling

**Database (`backend/database.py`, `backend/ai_database.py`):**
- Main database schema for users, strategies, trades, portfolios
- AI-specific schema for signals, predictions, ML models, feature data
- SQLite by default, PostgreSQL support for production

### Frontend Structure

**Main Application:**
- `App.jsx`: Entry point, renders ComprehensiveDashboard
- `main.jsx`: React app initialization
- Uses React 18 with Vite for fast development

**Component Organization:**
- `pages/`: Top-level pages (ComprehensiveDashboard)
- `components/ai/`: AI-specific components (DeepLearningDashboard, LSTM/Transformer trainers, EnsembleManager, FeatureExplorer, ModelManagement, PredictionVisualizer, SignalAggregator)
- `components/charts/`: Chart components (AdvancedPriceChart using Recharts)
- `components/common/`: Reusable UI components (Card, Loading, Section)
- `components/platform/`: Platform-specific components (PlatformMetrics)
- `services/`: API client (`api.js` for backend communication)

**Styling:**
- Tailwind CSS 3.4.13 for utility-first styling
- PostCSS for processing
- Responsive design patterns

### API Endpoints

**Core Endpoints (from `backend/app.py`):**
- `/symbols` - List available trading pairs
- `/price/{symbol}` - Current price for symbol
- `/candles/{symbol}?interval=1m&limit=200` - Historical candle data
- `/summary` - Market summary
- `/ab/status` - A/B testing status
- `/autotune/status` - Autotune status
- `/healthz`, `/health/quick`, `/health/live`, `/health/ready` - Health checks

**AI Endpoints (`/api/v1/ai/`):**
- TradingView webhook processing
- Signal aggregation and predictions
- Feature engineering
- Model training and management

**Deep Learning Endpoints (`/api/v1/deep-learning/`):**
- LSTM price prediction
- Transformer sequence forecasting
- Ensemble model aggregation

### Configuration & Environment

**Required Environment Variables:**
- `JWT_SECRET` - JWT signing key (32+ chars)
- `WEBHOOK_SECRET` - Webhook signature verification
- `ENCRYPTION_KEY` - Fernet encryption key (44 chars)
- `ALPACA_API_KEY`, `ALPACA_API_SECRET` - Trading broker credentials
- `ALPACA_BASE_URL` - Alpaca endpoint (paper-api or live)
- `SMTP_*` - Email configuration (optional)
- `DATABASE_PATH` - Database file location
- `CORS_ORIGINS` - Allowed CORS origins

**Generate Secrets:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"  # JWT/Webhook
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # Encryption
```

### Database Schema

**AI Database Tables:**
- `tradingview_signals`: Incoming TradingView webhook signals
- `ai_predictions`: AI model predictions with confidence scores
- `ml_models`: Trained model metadata and artifacts
- `signal_aggregation`: Multi-source signal aggregation
- `feature_store`: Engineered features for ML

**Main Database Tables:**
- Users, authentication, sessions
- Trading strategies and configurations
- Trade history and execution logs
- Portfolio positions and performance metrics
- Backup and reconciliation logs

## Development Notes

### Adding New Endpoints

1. Create router in appropriate module (`api_ai.py`, `api_deep_learning.py`, or new file)
2. Define Pydantic models for request/response validation
3. Import and include router in `backend/app.py` with error handling
4. Update frontend `services/api.js` with corresponding API calls

### Working with AI Models

- Models stored in database as serialized artifacts
- Feature engineering uses `backend/ai/features/feature_engineer.py`
- Signal aggregation combines TradingView, AI predictions, and RL agent signals
- Deep learning models use PyTorch (LSTM, Transformer) with training history tracking

### Testing Strategy

- **Unit tests** (`-m unit`): Individual component testing
- **Integration tests** (`-m integration`): API endpoint testing
- **E2E tests** (`-m e2e`): Full workflow testing
- **Trading tests** (`-m trading`): Broker integration (requires API keys)
- **WebSocket tests** (`-m websocket`): Real-time updates
- Coverage target: 80% (`pytest.ini` enforces this)

### Deployment Considerations

**Docker:**
- Multi-stage builds for minimal image size
- Health checks for automatic restart
- Persistent volumes for data (`/app/data`, `/app/backups`)
- Nginx reverse proxy in production profile
- Redis for session management (production)

**Kubernetes:**
- Namespace: `pinelab`
- HPA for auto-scaling based on CPU/memory
- PVC for persistent storage
- CronJobs for automated backups
- Ingress with cert-manager for TLS

### Common Patterns

**Database Operations:**
- Use context managers for SQLite connections
- Initialize schemas with `init_ai_schema()`, `init_db_schema()`
- Graceful fallback if AI modules not loaded

**Error Handling:**
- FastAPI HTTPException for API errors
- Middleware for centralized error handling
- Retry handler with exponential backoff for transient failures

**Async Operations:**
- Use `httpx.AsyncClient` for external API calls
- Background tasks for long-running operations
- WebSocket for real-time updates to frontend

**Security:**
- All secrets from environment variables
- Webhook signature verification
- Encryption for sensitive data (Fernet)
- JWT for authentication with 2FA support
