# PineScript Autogen Lab - Setup Guide

## 🎉 Welcome to Production-Ready Platform!

You now have a **production-grade trading automation platform** with:
- ✅ TradingView webhook execution
- ✅ Alpaca stock/ETF trading
- ✅ Order journal with CSV export
- ✅ Real P&L tracking
- ✅ HMAC signature authentication
- ✅ Position management
- ✅ Auto-optimization with Optuna
- ✅ A/B testing framework
- ✅ User authentication (JWT + API keys)
- ✅ Advanced logging & monitoring
- ✅ Email notifications
- ✅ Rate limiting & security

---

## 📋 Prerequisites

1. **Python 3.9+** installed
2. **Node.js 16+** and npm installed
3. **Alpaca account** (free paper trading account)
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Generate API keys (Paper Trading)

---

## 🚀 Installation

### Step 1: Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
notepad .env  # Windows
# nano .env   # Mac/Linux
```

**Required configuration:**

```env
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
WEBHOOK_SECRET=your_generated_secret_here

# Get from https://alpaca.markets (use Paper Trading keys for testing)
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_PAPER_TRADING=true

# Email notifications (optional - Phase 3)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@pinelab.com

# Logging (optional - Phase 3)
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Optional: Email Notifications Setup**

For Gmail:
1. Enable 2-factor authentication
2. Generate app password: https://myaccount.google.com/apppasswords
3. Add SMTP_USER and SMTP_PASSWORD to .env

For other providers (SendGrid, Mailgun):
- See [PHASE3.md](./PHASE3.md) for detailed setup

### Step 3: Start Backend

```bash
# From backend directory
uvicorn backend.app:app --reload --port 8080
```

You should see:
```
✅ Database initialized at ./data/pinelab.db
⚠️  WEBHOOK_SECRET not set... (if you haven't set it yet)
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Visit http://localhost:8080/docs to see the interactive API documentation.

### Step 4: Frontend Setup

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit http://localhost:5173 to see the dashboard.

---

## 🎯 Quick Start Guide

### 1. Configure Alpaca Broker

1. Go to http://localhost:5173
2. Click the **"Brokers"** tab
3. Click **"Add Broker"**
4. Enter your Alpaca API credentials
5. Keep "Paper Trading" checked for testing
6. Click **"Save Credentials"**

You should see your Alpaca account balance displayed!

### 2. Test Webhook Execution

Get a test signature:
```bash
curl http://localhost:8080/webhook/test-signature
```

This will show you an example curl command. Run it to execute a test order!

Example:
```bash
curl -X POST http://localhost:8080/exec \
  -H "Content-Type: application/json" \
  -H "X-Signature: abc123..." \
  -d '{"ticker":"AAPL","side":"buy","qty":1}'
```

### 3. View Your Orders

1. Check the **Dashboard** tab to see P&L summary
2. Go to http://localhost:8080/journal/orders to see order history
3. Download CSV: http://localhost:8080/journal/export/csv

---

## 📡 TradingView Integration

### Setup Webhook in TradingView

1. **Create an alert** on TradingView
2. In the **Notifications** section:
   - Check "Webhook URL"
   - Enter: `http://YOUR_SERVER:8080/exec`
   - (For local testing, use ngrok or similar tunnel)

3. **Set up signature** (optional but recommended):
   - In TradingView alert message, add custom header:
   ```json
   {
     "ticker": "{{ticker}}",
     "side": "buy",
     "qty": 10
   }
   ```

### Alert Message Templates

**Buy 10 shares:**
```json
{"ticker":"{{ticker}}","side":"buy","qty":10}
```

**Buy $1000 worth:**
```json
{"ticker":"{{ticker}}","side":"buy","notional":1000}
```

**Sell position:**
```json
{"ticker":"{{ticker}}","side":"sell","qty":10}
```

**Limit order:**
```json
{"ticker":"{{ticker}}","side":"buy","qty":10,"type":"limit","limit_price":{{close}}}
```

---

## 🔒 Security Setup (Production)

### Generate Webhook Secret

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output to your `.env` file as `WEBHOOK_SECRET=...`

### Configure TradingView Signature

To add signature verification in TradingView, you'll need to:
1. Set up a server with HTTPS (e.g., DigitalOcean, AWS)
2. Add the signature header using TradingView's webhook settings
3. Use a service like Cloudflare or nginx for SSL termination

**Note:** TradingView doesn't natively support custom headers. You may need a proxy service or custom implementation.

---

## 📊 API Endpoints

### Webhook Execution
- `POST /exec` - Execute order from webhook (requires signature)

### Orders & Journal
- `GET /journal/orders` - Get order history
- `GET /journal/orders/{id}` - Get specific order
- `GET /journal/export/csv` - Download orders as CSV

### P&L & Positions
- `GET /pnl/summary` - Get P&L metrics
- `GET /positions` - Get all open positions
- `POST /positions/sync` - Sync positions from Alpaca

### Broker Management
- `POST /broker/alpaca/set-creds` - Save Alpaca credentials
- `GET /broker/alpaca/account` - Get Alpaca account info
- `GET /broker/credentials` - List configured brokers

### Utilities
- `GET /healthz` - Health check
- `GET /webhook/test-signature` - Generate test signature
- `GET /docs` - Interactive API documentation

---

## 🧪 Testing

### Test Order Execution

```bash
# 1. Get test signature
curl http://localhost:8080/webhook/test-signature

# 2. Copy the curl example and run it

# 3. Check order status
curl http://localhost:8080/journal/orders
```

### Test with Different Order Types

```bash
# Market order
curl -X POST http://localhost:8080/exec \
  -H "Content-Type: application/json" \
  -H "X-Signature: YOUR_SIGNATURE" \
  -d '{"ticker":"SPY","side":"buy","qty":1}'

# Notional order ($100 worth)
curl -X POST http://localhost:8080/exec \
  -H "Content-Type: application/json" \
  -H "X-Signature: YOUR_SIGNATURE" \
  -d '{"ticker":"SPY","side":"buy","notional":100}'

# Limit order
curl -X POST http://localhost:8080/exec \
  -H "Content-Type: application/json" \
  -H "X-Signature: YOUR_SIGNATURE" \
  -d '{"ticker":"AAPL","side":"buy","qty":1,"type":"limit","limit_price":150.00}'
```

---

## 🐛 Troubleshooting

### Backend won't start

**Error:** `ModuleNotFoundError: No module named 'backend'`
```bash
# Make sure you're running from the project root:
cd /path/to/PinescriptAutogenLab
uvicorn backend.app:app --reload --port 8080
```

**Error:** `Database connection failed`
```bash
# Delete and reinitialize database
rm -rf data/
# Restart server - it will recreate the database
```

### Alpaca API errors

**Error:** `Alpaca not configured`
- Check that your API keys are in `.env`
- Verify keys are valid at alpaca.markets
- Ensure you're using Paper Trading keys initially

**Error:** `Account blocked` or `429 Too Many Requests`
- Wait a few minutes (rate limit)
- Check your Alpaca account status
- Verify you're not exceeding paper trading limits

### Frontend not loading

**Error:** `Failed to fetch`
- Verify backend is running on port 8080
- Check CORS configuration in backend/app.py
- Try http://localhost:8080/healthz in browser

---

## 📈 Next Steps

### Phase 1 Complete! ✅

You now have:
- Working webhook endpoint
- Alpaca integration
- Order tracking
- Real-time P&L

### All Phases Complete! ✅

**Phase 1:** Core trading platform ✓
**Phase 2:** Auto-optimization & A/B testing ✓
**Phase 3:** Production features (auth, monitoring, notifications) ✓

See comprehensive guides:
- [PHASE2.md](./PHASE2.md) - Optimization & A/B testing
- [PHASE3.md](./PHASE3.md) - Authentication & production features

### Test Phase 3 Features

**1. Register a User**
```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"testuser","password":"password123","full_name":"Test User"}'
```

**2. Login to Get JWT Token**
```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

**3. Access Protected Endpoint**
```bash
curl -X GET http://localhost:8080/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**4. View System Metrics**
```bash
curl http://localhost:8080/metrics
curl http://localhost:8080/health
```

---

## 💡 Tips & Best Practices

1. **Always use Paper Trading first** - Test thoroughly before live trading
2. **Monitor your orders** - Check the journal regularly
3. **Set position size limits** - Don't risk more than you can afford
4. **Keep webhooks secure** - Always use WEBHOOK_SECRET in production
5. **Back up your database** - `data/pinelab.db` contains all your orders
6. **Review P&L daily** - Track performance and adjust strategies

---

## 🆘 Support

- **API Documentation:** http://localhost:8080/docs
- **GitHub Issues:** Create an issue on the repository
- **Alpaca Support:** https://alpaca.markets/support

---

## 📝 License

This project is for educational and testing purposes. Use at your own risk. Always test with paper trading before using real money.

**Happy Trading! 🚀**
