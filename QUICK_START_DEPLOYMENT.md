# Quick Start Deployment Guide
## PinescriptAutogenLab AI Trading Platform

**Version:** 2.0.0
**Last Updated:** January 1, 2026

---

## Prerequisites

Before you begin, ensure you have installed:

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose** 2.0+ (usually included with Docker Desktop)
- **Git** (for cloning the repository)
- **Alpaca Account** (free paper trading: [alpaca.markets](https://alpaca.markets))

**System Requirements:**
- 4GB RAM minimum (8GB+ recommended for AI training)
- 10GB free disk space
- Windows 10/11, macOS, or Linux

---

## Option 1: Docker Compose Deployment (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PinescriptAutogenLab.git
cd PinescriptAutogenLab
```

### Step 2: Configure Environment Variables

The `.env` file is already generated with secure secrets. You only need to add your broker credentials:

```bash
# Edit .env file
notepad .env  # Windows
# or
nano .env     # Linux/Mac
```

**Required Configuration:**
```ini
# Update these lines with your Alpaca credentials:
ALPACA_API_KEY=your_alpaca_paper_api_key_here
ALPACA_API_SECRET=your_alpaca_paper_api_secret_here
```

**Optional Configuration:**
```ini
# Enable email notifications (optional):
SMTP_ENABLED=true
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password

# Update CORS for production (optional):
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Step 3: Build and Start Services

```bash
# Build images (first time only, takes 10-15 minutes)
docker compose -f docker-compose.prod.yml build

# Start all services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

### Step 4: Verify Deployment

1. **Backend API:** http://localhost:8000
   - Health check: http://localhost:8000/healthz
   - API docs: http://localhost:8000/docs

2. **Frontend UI:** http://localhost
   - Opens the AI Trading Dashboard

3. **Check service status:**
   ```bash
   docker compose -f docker-compose.prod.yml ps
   ```

### Step 5: First-Time Setup

1. **Access the Dashboard:** Open http://localhost in your browser

2. **Select a Symbol:** Choose a cryptocurrency (e.g., BTC_USDT) or stock (e.g., AAPL)

3. **Train Your First Model:**
   - Navigate to "Model Lab" tab
   - Click "Start Training"
   - Monitor training progress (takes 5-10 minutes for 50K timesteps)

4. **View Predictions:**
   - After training completes, check the "Predictions" panel
   - Predictions update every 15 seconds

---

## Option 2: Local Development Setup

### Backend Setup

```bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn backend.app:app --reload --port 8080
```

**Backend will be available at:** http://localhost:8080

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Frontend will be available at:** http://localhost:5173

---

## Option 3: Kubernetes Deployment

For production Kubernetes deployment, see `kubernetes/` directory.

```bash
# Create namespace
kubectl create namespace pinelab

# Apply configurations
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/backend-deployment.yaml
kubectl apply -f kubernetes/frontend-deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml

# Check deployment
kubectl get pods -n pinelab
kubectl get svc -n pinelab
```

---

## API Endpoints Overview

### Health & Status
- `GET /healthz` - Basic health check
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe

### AI/ML Endpoints
- `POST /api/v1/ai/model/train` - Train RL model
- `GET /api/v1/ai/model/predict/{ticker}` - Get predictions
- `POST /api/v1/ai/features/generate` - Generate ML features
- `GET /api/v1/ai/signal/aggregate/{ticker}` - Aggregate signals
- `POST /api/v1/ai/tradingview/webhook` - TradingView webhook
- `POST /api/v1/ai/chart/ohlcv` - Get OHLCV data
- `GET /api/v1/ai/chart/support-resistance/{ticker}` - S/R levels

### Trading Data
- `GET /symbols` - Get available symbols
- `GET /candles/{symbol}` - Get candlestick data

**Full API Documentation:** http://localhost:8000/docs

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server bind address |
| `PORT` | 8000 | Server port |
| `WORKERS` | 4 | Uvicorn worker processes |
| `JWT_SECRET` | (generated) | JWT signing secret |
| `WEBHOOK_SECRET` | (generated) | TradingView webhook secret |
| `ENCRYPTION_KEY` | (generated) | Data encryption key |
| `DATABASE_PATH` | /app/data/pinelab.db | SQLite database path |
| `LOG_LEVEL` | INFO | Logging level |
| `CORS_ORIGINS` | * | Allowed CORS origins |

### Docker Compose Profiles

```bash
# Start with Redis caching
docker compose -f docker-compose.prod.yml --profile production up -d

# Start with PostgreSQL database
docker compose -f docker-compose.prod.yml --profile postgres up -d

# Start with both
docker compose -f docker-compose.prod.yml --profile production --profile postgres up -d
```

---

## Common Operations

### View Logs

```bash
# All services
docker compose -f docker-compose.prod.yml logs -f

# Specific service
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
```

### Restart Services

```bash
# Restart all
docker compose -f docker-compose.prod.yml restart

# Restart backend only
docker compose -f docker-compose.prod.yml restart backend
```

### Stop Services

```bash
# Stop all services
docker compose -f docker-compose.prod.yml down

# Stop and remove volumes (data will be lost!)
docker compose -f docker-compose.prod.yml down -v
```

### Update and Rebuild

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

### Backup Data

```bash
# Backup database
docker cp pinelab-backend:/app/data/pinelab.db ./backup-$(date +%Y%m%d).db

# Backup all data
docker run --rm -v pinelab-data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data
```

---

## Troubleshooting

### Issue: "Port already in use"

**Error:** `Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use`

**Solutions:**
```bash
# Option 1: Stop the conflicting process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9

# Option 2: Change port in docker-compose.prod.yml
ports:
  - "8001:8000"  # Change 8000 to 8001
```

### Issue: "Backend fails to start"

**Check logs:**
```bash
docker compose -f docker-compose.prod.yml logs backend
```

**Common causes:**
1. **Missing dependencies:** Rebuild image
   ```bash
   docker compose -f docker-compose.prod.yml build --no-cache backend
   ```

2. **Database locked:** Remove lock file
   ```bash
   docker compose -f docker-compose.prod.yml exec backend rm /app/data/pinelab.db-journal
   ```

3. **Permission issues:** Fix volume permissions
   ```bash
   docker compose -f docker-compose.prod.yml exec backend chown -R pinelab:pinelab /app/data
   ```

### Issue: "Frontend shows 'Network Error'"

**Causes:**
1. **Backend not running:** Check `docker compose ps`
2. **CORS error:** Update `CORS_ORIGINS` in `.env`
3. **Wrong API URL:** Check `frontend/.env` has `VITE_API_URL=http://localhost:8000`

**Fix:**
```bash
# Rebuild frontend with correct API URL
docker compose -f docker-compose.prod.yml build frontend
docker compose -f docker-compose.prod.yml up -d frontend
```

### Issue: "AI features not working"

**Error:** `WARNING: AI endpoints not loaded`

**Cause:** Missing AI dependencies

**Fix:**
```bash
# Rebuild backend image
docker compose -f docker-compose.prod.yml build --no-cache backend
docker compose -f docker-compose.prod.yml up -d backend

# Verify AI endpoints are loaded
docker compose -f docker-compose.prod.yml logs backend | grep "AI Trading endpoints"
# Should see: "✅ AI Trading endpoints loaded successfully"
```

### Issue: "Model training fails"

**Common causes:**
1. **Insufficient memory:** Increase Docker memory limit (Settings → Resources)
2. **Invalid symbol:** Use supported symbols (BTC_USDT, ETH_USDT, AAPL, etc.)
3. **No market data:** Check external API connectivity

**Debug:**
```bash
# Check available memory
docker stats

# Test data fetching
curl "http://localhost:8000/api/v1/ai/chart/ohlcv" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC_USDT","timeframe":"1h","limit":1000}'
```

### Issue: "Database errors"

**Error:** `database is locked`

**Fix:**
```bash
# Stop all services
docker compose -f docker-compose.prod.yml down

# Remove database lock
docker volume inspect pinelab-data
# Note the Mountpoint path
sudo rm <mountpoint>/pinelab.db-journal

# Restart
docker compose -f docker-compose.prod.yml up -d
```

---

## Performance Tuning

### For Development (Low Resources)

Edit `docker-compose.prod.yml`:
```yaml
backend:
  environment:
    - WORKERS=2  # Reduce from 4
```

### For Production (High Performance)

```yaml
backend:
  environment:
    - WORKERS=8  # Increase workers
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
```

Enable Redis caching:
```bash
docker compose -f docker-compose.prod.yml --profile production up -d
```

### GPU Support (for faster training)

Requires NVIDIA GPU and nvidia-docker.

Add to `docker-compose.prod.yml`:
```yaml
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## Security Checklist

### Before Production Deployment:

- [ ] Change all default secrets in `.env`
- [ ] Set strong `JWT_SECRET`, `WEBHOOK_SECRET`, `ENCRYPTION_KEY`
- [ ] Configure Alpaca API credentials
- [ ] Set production `CORS_ORIGINS` (remove `*`)
- [ ] Enable HTTPS (use reverse proxy like Nginx)
- [ ] Set `DEBUG=false`
- [ ] Configure firewall rules
- [ ] Set up automated backups
- [ ] Enable monitoring and alerts
- [ ] Review and restrict exposed ports
- [ ] Use PostgreSQL instead of SQLite
- [ ] Enable rate limiting
- [ ] Set up log aggregation

---

## Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Frontend health
curl http://localhost/

# Service status
docker compose -f docker-compose.prod.yml ps
```

### Metrics

Metrics are exposed on port 9090:
```bash
curl http://localhost:9090/metrics
```

**Prometheus scrape config:**
```yaml
scrape_configs:
  - job_name: 'pinelab-backend'
    static_configs:
      - targets: ['localhost:9090']
```

### Logs

```bash
# Real-time logs
docker compose -f docker-compose.prod.yml logs -f

# Last 100 lines
docker compose -f docker-compose.prod.yml logs --tail=100

# Search logs
docker compose -f docker-compose.prod.yml logs | grep ERROR

# Export logs
docker compose -f docker-compose.prod.yml logs > pinelab-logs-$(date +%Y%m%d).log
```

---

## Next Steps

1. **Train Production Models:** Use real market data to train models
2. **Set Up Monitoring:** Configure Prometheus + Grafana
3. **Enable Notifications:** Configure SMTP for email alerts
4. **Implement Testing:** Add unit and integration tests
5. **Plan Phase 2:** Review `AI_TRADING_PLATFORM_PLAN.md` for upcoming features
6. **Join Community:** Share feedback and contribute

---

## Additional Resources

- **Full Documentation:** See `README.md`
- **API Reference:** http://localhost:8000/docs
- **Architecture Guide:** See `IMPLEMENTATION_SUMMARY.md`
- **Production Assessment:** See `PRODUCTION_READINESS_ASSESSMENT.md`
- **AI Implementation:** See `AI_IMPLEMENTATION_GUIDE.md`

---

## Getting Help

**Issues or Questions?**
- Check the troubleshooting section above
- Review logs: `docker compose logs -f`
- Open an issue on GitHub
- Contact support

---

**Happy Trading! 🚀**
