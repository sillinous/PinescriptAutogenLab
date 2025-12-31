# PinescriptAutogenLab - Quick Start Guide

Complete guide to run the application with frontend in development and production modes.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Mode](#development-mode-local)
3. [Production Mode - Docker](#production-mode-docker)
4. [Production Mode - Kubernetes](#production-mode-kubernetes)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Node.js 18+** (for frontend)
- **Python 3.11+** (for backend)
- **Docker & Docker Compose** (for containerized deployment)
- **Git** (for version control)

### Install Node.js (if not installed)

**Windows**:
```powershell
# Download from https://nodejs.org/
# Or using Chocolatey:
choco install nodejs
```

**Mac**:
```bash
brew install node
```

**Linux**:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

---

## Development Mode (Local)

### Option 1: Run Frontend + Backend Separately

This is the **recommended approach for development** as it provides hot-reload for both frontend and backend.

#### Step 1: Start Backend (already running)

```bash
# In terminal 1 (backend)
cd C:\GitHub\GitHubRoot\sillinous\PinescriptAutogenLab
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Backend will be at**: http://localhost:8000

#### Step 2: Start Frontend

```bash
# In terminal 2 (frontend)
cd C:\GitHub\GitHubRoot\sillinous\PinescriptAutogenLab\frontend

# Install dependencies (if not done)
npm install

# Start development server
npm run dev
```

**Frontend will be at**: http://localhost:5173 (or http://localhost:3000)

#### Step 3: Open Application

1. Open browser to **http://localhost:5173**
2. The frontend will automatically connect to backend at `http://localhost:8000`

### Features Available in Dev Mode

- ✅ **Hot Reload**: Changes to code automatically refresh
- ✅ **Source Maps**: Easy debugging
- ✅ **Console Logs**: Full error messages
- ✅ **React DevTools**: Component inspection
- ✅ **Fast Iteration**: No build step needed

---

## Production Mode - Docker

### Option 2: Docker Compose (Recommended for Production)

This runs both frontend and backend in containers with production optimizations.

#### Step 1: Configure Environment

```bash
# Copy environment template
cp .env.docker .env

# Edit .env with your actual values
nano .env  # or use your text editor
```

**Required Configuration**:
```env
# Generate these secrets first!
JWT_SECRET=your_jwt_secret_32_chars_minimum
WEBHOOK_SECRET=your_webhook_secret
ENCRYPTION_KEY=your_fernet_key_44_chars

# Alpaca credentials (get from alpaca.markets)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# SMTP for emails (optional but recommended)
SMTP_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password
```

**Generate Secrets**:
```bash
# JWT Secret
python -c "import secrets; print(secrets.token_hex(32))"

# Webhook Secret
python -c "import secrets; print(secrets.token_hex(32))"

# Encryption Key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

#### Step 2: Run with Docker Compose

```bash
# Development mode (SQLite, no Nginx)
docker-compose up -d

# Production mode (with Nginx, Redis, optional PostgreSQL)
docker-compose --profile production up -d
```

#### Step 3: Access Application

**Development Mode**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Production Mode** (with Nginx):
- Application: http://localhost
- Backend API: http://localhost/api
- WebSocket: ws://localhost/ws

#### Step 4: View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f nginx
```

#### Step 5: Stop Services

```bash
# Stop but keep data
docker-compose down

# Stop and remove all data
docker-compose down -v
```

### Docker Production Features

- ✅ **Optimized Builds**: Multi-stage builds, minimal image size
- ✅ **Auto-restart**: Services restart on failure
- ✅ **Health Checks**: Automatic service monitoring
- ✅ **Persistent Storage**: Data survives restarts
- ✅ **Nginx Reverse Proxy**: Load balancing, SSL termination
- ✅ **Redis Caching**: Fast session management
- ✅ **Auto-scaling**: Can scale backend replicas

---

## Production Mode - Kubernetes

### Option 3: Kubernetes Deployment

For large-scale production with auto-scaling, high availability, and enterprise features.

#### Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or local with Minikube)
- kubectl configured
- Container registry (Docker Hub, GCR, ECR, etc.)

#### Step 1: Build and Push Images

```bash
# Build images
docker build -t your-registry/pinelab-backend:latest .
docker build -t your-registry/pinelab-frontend:latest ./frontend

# Push to registry
docker push your-registry/pinelab-backend:latest
docker push your-registry/pinelab-frontend:latest
```

#### Step 2: Update Kubernetes Manifests

Edit `k8s/deployment.yaml` and update image references:
```yaml
image: your-registry/pinelab-backend:latest  # Change this
```

#### Step 3: Create Secrets

```bash
# Generate secrets
JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
WEBHOOK_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Create Kubernetes secret
kubectl create secret generic pinelab-secrets \
  --from-literal=JWT_SECRET="$JWT_SECRET" \
  --from-literal=WEBHOOK_SECRET="$WEBHOOK_SECRET" \
  --from-literal=ENCRYPTION_KEY="$ENCRYPTION_KEY" \
  --from-literal=ALPACA_API_KEY="your_alpaca_key" \
  --from-literal=ALPACA_API_SECRET="your_alpaca_secret" \
  --from-literal=SMTP_USERNAME="your_email@gmail.com" \
  --from-literal=SMTP_PASSWORD="your_password" \
  --namespace=pinelab
```

#### Step 4: Deploy to Kubernetes

```bash
# Deploy everything
kubectl apply -f k8s/

# Check status
kubectl get pods -n pinelab
kubectl get svc -n pinelab
kubectl get ingress -n pinelab
```

#### Step 5: Access Application

```bash
# Get ingress IP
kubectl get ingress -n pinelab

# Access at your configured domain
# https://pinelab.yourdomain.com
```

### Kubernetes Production Features

- ✅ **Auto-scaling**: HPA scales based on CPU/memory
- ✅ **High Availability**: Multiple replicas with load balancing
- ✅ **Rolling Updates**: Zero-downtime deployments
- ✅ **Health Monitoring**: Liveness and readiness probes
- ✅ **Automatic Backups**: CronJob for daily backups
- ✅ **SSL/TLS**: Automatic certificates with cert-manager
- ✅ **Persistent Storage**: PVCs for data persistence

---

## Configuration

### Environment Variables

#### Backend Configuration

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `JWT_SECRET` | Yes | JWT signing key | `secrets.token_hex(32)` |
| `WEBHOOK_SECRET` | Yes | Webhook signature key | `secrets.token_hex(32)` |
| `ENCRYPTION_KEY` | Yes | Fernet encryption key | `Fernet.generate_key()` |
| `ALPACA_API_KEY` | Yes* | Alpaca API key | From alpaca.markets |
| `ALPACA_API_SECRET` | Yes* | Alpaca API secret | From alpaca.markets |
| `ALPACA_BASE_URL` | No | Alpaca endpoint | `paper-api.alpaca.markets` |
| `SMTP_ENABLED` | No | Enable email | `true` |
| `SMTP_HOST` | No | SMTP server | `smtp.gmail.com` |
| `SMTP_PORT` | No | SMTP port | `587` |
| `SMTP_USERNAME` | No | Email address | `you@gmail.com` |
| `SMTP_PASSWORD` | No | App password | From Gmail settings |
| `DATABASE_PATH` | No | Database location | `data/pinelab.db` |
| `CORS_ORIGINS` | No | Allowed origins | `http://localhost:3000` |

*Required for trading functionality

#### Frontend Configuration

Create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

For production:
```env
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=wss://api.yourdomain.com
```

### CORS Configuration

For local development, CORS is already configured to allow:
- `http://localhost:3000`
- `http://localhost:5173`
- `http://localhost:8000`

For production, update `.env`:
```env
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### SSL/TLS Setup

#### Development (Self-Signed)

Already configured in `nginx/ssl/` for local testing.

#### Production (Let's Encrypt)

**Using cert-manager on Kubernetes**:
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Certificate will be automatically provisioned via ingress annotations
```

**Using Docker with Nginx**:
```bash
# Use Certbot
docker run -it --rm --name certbot \
  -v "/etc/letsencrypt:/etc/letsencrypt" \
  -v "/var/lib/letsencrypt:/var/lib/letsencrypt" \
  certbot/certbot certonly --standalone -d yourdomain.com
```

---

## Troubleshooting

### Frontend Issues

#### "Cannot connect to backend"

**Solution**:
```bash
# Check backend is running
curl http://localhost:8000/health/quick

# Check CORS configuration in backend/.env
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Restart backend after changes
```

#### "npm install fails"

**Solution**:
```bash
# Clear cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### "Port 5173 already in use"

**Solution**:
```bash
# Kill process on port 5173
# Windows:
netstat -ano | findstr :5173
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5173 | xargs kill -9

# Or use different port
npm run dev -- --port 3001
```

### Backend Issues

#### "Database locked"

**Solution**:
```bash
# Stop all instances
pkill -f uvicorn

# Delete lock file
rm data/pinelab.db-journal

# Restart
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

#### "ModuleNotFoundError"

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific module
pip install optuna pyotp qrcode psutil
```

### Docker Issues

#### "Port 8000 already in use"

**Solution**:
```bash
# Check running containers
docker ps

# Stop conflicting service
docker-compose down

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Map to different port
```

#### "Cannot build image"

**Solution**:
```bash
# Clear Docker cache
docker builder prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### Kubernetes Issues

#### "Pods in CrashLoopBackOff"

**Solution**:
```bash
# Check logs
kubectl logs <pod-name> -n pinelab

# Check events
kubectl describe pod <pod-name> -n pinelab

# Common fixes:
# - Update secrets
# - Check resource limits
# - Verify PVC exists
```

#### "Ingress not working"

**Solution**:
```bash
# Check ingress controller is installed
kubectl get pods -n ingress-nginx

# Verify ingress resource
kubectl describe ingress pinelab-ingress -n pinelab

# Check DNS points to ingress IP
kubectl get ingress -n pinelab
```

---

## Performance Optimization

### Frontend

```javascript
// frontend/vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts']
        }
      }
    }
  }
})
```

### Backend

```python
# Increase worker count for production
uvicorn backend.app:app --workers 4 --host 0.0.0.0 --port 8000
```

### Database

```bash
# Use PostgreSQL for production instead of SQLite
DATABASE_URL=postgresql://user:pass@localhost:5432/pinelab
```

---

## Monitoring

### View Application Logs

**Docker**:
```bash
docker-compose logs -f backend
```

**Kubernetes**:
```bash
kubectl logs -f deployment/pinelab-backend -n pinelab
```

### Monitor Resources

**Docker**:
```bash
docker stats
```

**Kubernetes**:
```bash
kubectl top pods -n pinelab
kubectl top nodes
```

### Health Checks

- Liveness: http://localhost:8000/health/live
- Readiness: http://localhost:8000/health/ready
- Full: http://localhost:8000/health

---

## Next Steps

1. ✅ **Start Development**: Run frontend + backend locally
2. ✅ **Configure Secrets**: Set up API keys and encryption
3. ✅ **Test Features**: Try all endpoints via Swagger UI
4. ✅ **Deploy Docker**: Use docker-compose for staging
5. ✅ **Setup Monitoring**: Configure health checks and alerts
6. ✅ **Production Deploy**: Deploy to Kubernetes
7. ✅ **Setup CI/CD**: Automate testing and deployment

---

## Quick Reference Commands

### Development
```bash
# Backend
python -m uvicorn backend.app:app --reload

# Frontend
cd frontend && npm run dev
```

### Docker
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f
```

### Kubernetes
```bash
# Deploy
kubectl apply -f k8s/

# Status
kubectl get all -n pinelab

# Logs
kubectl logs -f deployment/pinelab-backend -n pinelab
```

---

For more details, see:
- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [TESTING.md](TESTING.md) - Testing guide
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Project overview
