# PinescriptAutogenLab Deployment Guide

Comprehensive guide for deploying PinescriptAutogenLab in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Checklist](#production-checklist)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Python 3.11+**
- **Docker 20.10+** (for containerized deployment)
- **Docker Compose 2.0+**
- **PostgreSQL 13+** or **SQLite** (default)

### Optional

- **Kubernetes 1.23+** (for K8s deployment)
- **kubectl** (configured)
- **Nginx Ingress Controller**
- **cert-manager** (for TLS)

### API Keys & Credentials

1. **Alpaca Trading API**
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Get paper trading credentials

2. **SMTP Credentials** (for emails)
   - Gmail with App Password (recommended)
   - Or any SMTP provider

3. **Encryption Keys**
   ```bash
   # Generate JWT secret
   openssl rand -hex 32

   # Generate Webhook secret
   openssl rand -hex 32

   # Generate Encryption key (Fernet)
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/PinescriptAutogenLab.git
cd PinescriptAutogenLab
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

**Required Variables:**
```env
JWT_SECRET=your_jwt_secret_32_chars_minimum
WEBHOOK_SECRET=your_webhook_secret
ENCRYPTION_KEY=your_fernet_key_44_chars
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
```

### 5. Initialize Database

```bash
python -c "from backend.database import init_db; init_db()"
```

### 6. Run Server

```bash
# Development server
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Or with more workers
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7. Access Application

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Docker Deployment

### Quick Start

```bash
# Copy environment file
cp .env.docker .env

# Edit configuration
nano .env

# Run deployment script
chmod +x scripts/deploy.sh
./scripts/deploy.sh development
```

### Manual Deployment

#### 1. Build Images

```bash
# Build backend
docker build -t pinelab/backend:latest .

# Build frontend
cd frontend
docker build -t pinelab/frontend:latest .
cd ..
```

#### 2. Start Services

```bash
# Development mode
docker-compose up -d

# Production mode (with Nginx, Redis)
docker-compose --profile production up -d
```

#### 3. Verify Deployment

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs -f backend

# Test health
curl http://localhost:8000/health
```

### Docker Compose Profiles

#### Development Profile (Default)

```bash
docker-compose up -d
```

**Includes:**
- Backend API
- Frontend
- SQLite database

#### Production Profile

```bash
docker-compose --profile production up -d
```

**Includes:**
- Backend API (multiple replicas)
- Frontend
- Nginx reverse proxy
- Redis (caching)
- PostgreSQL (optional)

### Useful Docker Commands

```bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Scale backend
docker-compose up -d --scale backend=3

# Execute command in container
docker-compose exec backend python -c "from backend.database import init_db; init_db()"

# Access container shell
docker-compose exec backend /bin/bash
```

---

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes Cluster**
   - Local: Minikube, Kind, Docker Desktop
   - Cloud: GKE, EKS, AKS

2. **Nginx Ingress Controller**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
   ```

3. **cert-manager** (Optional, for TLS)
   ```bash
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   ```

### Deployment Steps

#### 1. Build and Push Images

```bash
# Build images
docker build -t your-registry/pinelab-backend:latest .
docker build -t your-registry/pinelab-frontend:latest ./frontend

# Push to registry
docker push your-registry/pinelab-backend:latest
docker push your-registry/pinelab-frontend:latest

# Update image references in k8s/deployment.yaml
```

#### 2. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

#### 3. Configure Secrets

```bash
# Generate secrets
JWT_SECRET=$(openssl rand -hex 32)
WEBHOOK_SECRET=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Create secret
kubectl create secret generic pinelab-secrets \
  --from-literal=JWT_SECRET="$JWT_SECRET" \
  --from-literal=WEBHOOK_SECRET="$WEBHOOK_SECRET" \
  --from-literal=ENCRYPTION_KEY="$ENCRYPTION_KEY" \
  --from-literal=ALPACA_API_KEY="your_key" \
  --from-literal=ALPACA_API_SECRET="your_secret" \
  --from-literal=SMTP_USERNAME="your_email@gmail.com" \
  --from-literal=SMTP_PASSWORD="your_password" \
  --namespace=pinelab
```

#### 4. Apply ConfigMap

```bash
# Edit k8s/configmap.yaml if needed
kubectl apply -f k8s/configmap.yaml
```

#### 5. Create Persistent Volumes

```bash
kubectl apply -f k8s/pvc.yaml
```

#### 6. Deploy Application

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

#### 7. Configure Ingress

```bash
# Edit k8s/ingress.yaml with your domain
kubectl apply -f k8s/ingress.yaml
```

#### 8. Enable Auto-scaling (Optional)

```bash
kubectl apply -f k8s/hpa.yaml
```

#### 9. Schedule CronJobs (Optional)

```bash
kubectl apply -f k8s/cronjob.yaml
```

### Verification

```bash
# Check pods
kubectl get pods -n pinelab

# Check services
kubectl get svc -n pinelab

# Check ingress
kubectl get ingress -n pinelab

# View logs
kubectl logs -f deployment/pinelab-backend -n pinelab

# Port forward for testing
kubectl port-forward svc/pinelab-backend 8000:8000 -n pinelab
```

---

## Production Checklist

### Security

- [ ] **Use production-grade secrets management**
  - Vault, AWS Secrets Manager, Google Secret Manager
  - Never commit secrets to git

- [ ] **Configure TLS/SSL certificates**
  - Use cert-manager with Let's Encrypt
  - Or upload custom certificates

- [ ] **Enable CORS properly**
  - Set CORS_ORIGINS to actual frontend domain
  - Don't use wildcard (*) in production

- [ ] **Configure rate limiting**
  - Webhook endpoints: 100 req/min
  - API endpoints: 60 req/min
  - Adjust based on needs

- [ ] **Enable audit logging**
  - Set ENABLE_AUDIT_LOG=true
  - Monitor audit logs regularly

- [ ] **Configure 2FA**
  - Set ENABLE_2FA=true
  - Require for admin accounts

- [ ] **Enable email verification**
  - Set ENABLE_EMAIL_VERIFICATION=true

### Reliability

- [ ] **Set up automated backups**
  - Enable AUTO_BACKUP_ON_SHUTDOWN=true
  - Configure CronJob for daily backups
  - Test restore procedure

- [ ] **Configure health checks**
  - Liveness probe: /health/live
  - Readiness probe: /health/ready
  - Startup probe for slow starts

- [ ] **Enable reconciliation**
  - Set RECONCILIATION_INTERVAL_MINUTES=5
  - Monitor reconciliation logs

- [ ] **Configure retries**
  - RETRY_MAX_ATTEMPTS=3
  - RETRY_BASE_DELAY=1.0
  - RETRY_MAX_DELAY=60.0

### Monitoring

- [ ] **Set up application monitoring**
  - Prometheus metrics
  - Grafana dashboards
  - Alert rules

- [ ] **Configure log aggregation**
  - ELK Stack or Loki
  - Centralized logging
  - Log retention policy

- [ ] **Enable alerting**
  - Order failures
  - System errors
  - Health check failures
  - High resource usage

- [ ] **Monitor key metrics**
  - Request latency
  - Error rates
  - Order success rate
  - Database size
  - Backup status

### Performance

- [ ] **Set resource limits**
  - CPU: 250m-1000m (requests-limits)
  - Memory: 512Mi-2Gi
  - Adjust based on load

- [ ] **Configure auto-scaling**
  - HPA based on CPU/memory
  - Min replicas: 3
  - Max replicas: 10

- [ ] **Enable caching**
  - Redis for session management
  - API response caching
  - Static asset caching

- [ ] **Optimize database**
  - Use PostgreSQL for production
  - Configure connection pooling
  - Regular vacuum/analyze

### Deployment

- [ ] **Use CI/CD pipeline**
  - Automated testing
  - Automated deployment
  - Rollback capability

- [ ] **Configure environment**
  - ENVIRONMENT=production
  - DEBUG=false
  - LOG_LEVEL=INFO (or WARNING)

- [ ] **Test disaster recovery**
  - Backup restore procedure
  - Failover testing
  - Data recovery

- [ ] **Document runbooks**
  - Deployment procedure
  - Rollback procedure
  - Incident response

---

## Monitoring & Maintenance

### Daily Tasks

```bash
# Check service health
curl https://api.pinelab.example.com/health

# View recent logs
kubectl logs --tail=100 -l app=pinelab -n pinelab

# Check pod status
kubectl get pods -n pinelab
```

### Weekly Tasks

```bash
# Review metrics
kubectl top pods -n pinelab

# Check backup status
kubectl get cronjobs -n pinelab

# Review audit logs
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://api.pinelab.example.com/admin/audit-log
```

### Monthly Tasks

- Review and update dependencies
- Check and rotate secrets
- Review resource usage and scale as needed
- Test backup restore procedure
- Review and update documentation

### Backup Management

```bash
# Manual backup
kubectl exec -it deployment/pinelab-backend -n pinelab -- \
  python -c "from backend.reliability.backup_service import get_backup_service; \
  service = get_backup_service(); \
  result = service.create_backup(compress=True, encrypt=True); \
  print(result)"

# List backups
kubectl exec deployment/pinelab-backend -n pinelab -- \
  python -c "from backend.reliability.backup_service import get_backup_service; \
  service = get_backup_service(); \
  backups = service.list_backups(); \
  for b in backups: print(b['name'])"

# Download backups
POD=$(kubectl get pod -n pinelab -l app=pinelab,component=backend -o jsonpath='{.items[0].metadata.name}')
kubectl cp pinelab/$POD:/app/backups ./backups
```

---

## Troubleshooting

### Backend Not Starting

**Symptoms:** Pod in CrashLoopBackOff, service unavailable

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n pinelab
kubectl logs <pod-name> -n pinelab
```

**Common Causes:**
1. **Missing secrets** - Check secrets exist: `kubectl get secrets -n pinelab`
2. **Database issues** - Check PVC: `kubectl get pvc -n pinelab`
3. **Configuration errors** - Validate configmap: `kubectl describe cm pinelab-config -n pinelab`

### High Latency

**Symptoms:** Slow API responses

**Diagnosis:**
```bash
kubectl top pods -n pinelab
kubectl describe hpa -n pinelab
```

**Solutions:**
1. Scale up replicas
2. Increase resource limits
3. Enable caching
4. Optimize database queries

### Database Connection Errors

**Symptoms:** "Database connection failed" in logs

**Solutions:**
1. Check PVC status: `kubectl get pvc -n pinelab`
2. Verify volume mounts: `kubectl describe pod <pod-name> -n pinelab`
3. Check database file permissions
4. Verify DATABASE_PATH in configmap

### Webhook Signature Failures

**Symptoms:** 401 Unauthorized on webhook endpoint

**Solutions:**
1. Verify WEBHOOK_SECRET matches TradingView
2. Check webhook payload format
3. Review webhook logs: `kubectl logs -f deployment/pinelab-backend -n pinelab | grep webhook`

### SSL/TLS Certificate Issues

**Symptoms:** Certificate errors, HTTPS not working

**Solutions:**
1. Check cert-manager logs: `kubectl logs -n cert-manager deployment/cert-manager`
2. Verify certificate: `kubectl describe certificate -n pinelab`
3. Check ingress annotations
4. Ensure DNS points to ingress IP

### Out of Disk Space

**Symptoms:** "No space left on device" errors

**Solutions:**
1. Check PVC usage
2. Cleanup old backups: Run cleanup CronJob manually
3. Increase PVC size:
   ```bash
   kubectl edit pvc pinelab-data-pvc -n pinelab
   # Increase storage request
   ```
4. Review log retention settings

---

## Additional Resources

- [Backend API Documentation](http://localhost:8000/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Alpaca API Documentation](https://alpaca.markets/docs/)

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/PinescriptAutogenLab/issues
- Documentation: See FEATURES_ADDED.md and GAP_RESOLUTION_REPORT.md
