# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying PinescriptAutogenLab to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (1.23+)
- kubectl configured to access your cluster
- Nginx Ingress Controller installed
- cert-manager (optional, for automatic TLS certificates)
- Persistent storage provisioner

## Quick Start

### 1. Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### 2. Configure Secrets

**IMPORTANT**: Update `secret.yaml` with your actual secrets before applying.

```bash
# Generate secrets
JWT_SECRET=$(openssl rand -hex 32)
WEBHOOK_SECRET=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Create secret manually (recommended)
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

### 3. Apply ConfigMap

```bash
kubectl apply -f configmap.yaml
```

### 4. Create Persistent Volumes

```bash
kubectl apply -f pvc.yaml
```

### 5. Deploy Application

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 6. Configure Ingress

Update `ingress.yaml` with your domain name, then:

```bash
kubectl apply -f ingress.yaml
```

### 7. Configure Autoscaling (Optional)

```bash
kubectl apply -f hpa.yaml
```

### 8. Schedule CronJobs (Optional)

```bash
kubectl apply -f cronjob.yaml
```

## Deployment Order

Deploy in this order:

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
# Create secrets manually (see above)
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
kubectl apply -f cronjob.yaml
```

Or use a single command:

```bash
kubectl apply -f k8s/
```

## Verification

### Check Pods

```bash
kubectl get pods -n pinelab
```

### Check Services

```bash
kubectl get svc -n pinelab
```

### Check Ingress

```bash
kubectl get ingress -n pinelab
```

### View Logs

```bash
# Backend logs
kubectl logs -f deployment/pinelab-backend -n pinelab

# Frontend logs
kubectl logs -f deployment/pinelab-frontend -n pinelab
```

### Test Health

```bash
# Port forward
kubectl port-forward svc/pinelab-backend 8000:8000 -n pinelab

# Test health endpoint
curl http://localhost:8000/health
```

## Scaling

### Manual Scaling

```bash
# Scale backend
kubectl scale deployment pinelab-backend --replicas=5 -n pinelab

# Scale frontend
kubectl scale deployment pinelab-frontend --replicas=3 -n pinelab
```

### Auto-scaling

Horizontal Pod Autoscaler (HPA) is configured in `hpa.yaml` and will automatically scale based on CPU/memory usage.

## Updates and Rollbacks

### Update Deployment

```bash
# Update image
kubectl set image deployment/pinelab-backend backend=pinelab/backend:v2.0.0 -n pinelab

# Roll out restart
kubectl rollout restart deployment/pinelab-backend -n pinelab
```

### Check Rollout Status

```bash
kubectl rollout status deployment/pinelab-backend -n pinelab
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/pinelab-backend -n pinelab

# Rollback to specific revision
kubectl rollout undo deployment/pinelab-backend --to-revision=2 -n pinelab
```

## Backup and Restore

### Manual Backup

```bash
kubectl exec -it deployment/pinelab-backend -n pinelab -- \
  python -c "from backend.reliability.backup_service import get_backup_service; \
  service = get_backup_service(); \
  result = service.create_backup(compress=True, encrypt=True); \
  print(result)"
```

### Access Backup Files

```bash
# Get pod name
POD=$(kubectl get pod -n pinelab -l app=pinelab,component=backend -o jsonpath='{.items[0].metadata.name}')

# Copy backup to local
kubectl cp pinelab/$POD:/app/backups ./backups
```

## Monitoring

### Resource Usage

```bash
kubectl top pods -n pinelab
kubectl top nodes
```

### Events

```bash
kubectl get events -n pinelab --sort-by='.lastTimestamp'
```

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod <pod-name> -n pinelab
kubectl logs <pod-name> -n pinelab
```

### Service Not Accessible

```bash
kubectl get endpoints -n pinelab
kubectl describe svc pinelab-backend -n pinelab
```

### Ingress Issues

```bash
kubectl describe ingress pinelab-ingress -n pinelab
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

### Database Connection Issues

```bash
# Check PVC
kubectl get pvc -n pinelab

# Check volume mounts
kubectl describe pod <pod-name> -n pinelab
```

## Production Checklist

- [ ] Use production-grade secrets management (Vault, AWS Secrets Manager, etc.)
- [ ] Configure TLS certificates (cert-manager with Let's Encrypt)
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure log aggregation (ELK, Loki)
- [ ] Set resource requests and limits
- [ ] Configure network policies
- [ ] Set up backup automation
- [ ] Configure alerting
- [ ] Implement pod disruption budgets
- [ ] Use multiple availability zones
- [ ] Configure RBAC properly
- [ ] Enable audit logging
- [ ] Set up disaster recovery plan

## Clean Up

```bash
# Delete all resources
kubectl delete namespace pinelab

# Or delete individually
kubectl delete -f k8s/
```

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Nginx Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager](https://cert-manager.io/)
- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
