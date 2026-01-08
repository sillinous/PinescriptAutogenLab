/**
 * API Service Layer
 * Centralized API calls for the AI Trading Platform
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class APIError extends Error {
  constructor(message, status, data) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.data = data
  }
}

async function fetchJSON(url, options = {}) {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    const data = await response.json()

    if (!response.ok) {
      throw new APIError(
        data.detail || data.message || 'API request failed',
        response.status,
        data
      )
    }

    return data
  } catch (error) {
    if (error instanceof APIError) throw error
    throw new APIError(error.message, 0, null)
  }
}

// ============================================================================
// Trading Data APIs
// ============================================================================

export const tradingAPI = {
  async getSymbols() {
    return fetchJSON(`${API_BASE}/symbols`)
  },

  async getPrice(symbol) {
    return fetchJSON(`${API_BASE}/price/${symbol}`)
  },

  async getCandles(symbol, interval = '1h', limit = 200) {
    return fetchJSON(`${API_BASE}/candles/${symbol}?interval=${interval}&limit=${limit}`)
  },

  async getABStatus(symbolA = null, symbolB = null) {
    let url = `${API_BASE}/ab/status`
    const params = []
    if (symbolA) params.push(`symbol_a=${encodeURIComponent(symbolA)}`)
    if (symbolB) params.push(`symbol_b=${encodeURIComponent(symbolB)}`)
    if (params.length > 0) url += `?${params.join('&')}`
    return fetchJSON(url)
  },

  async getAutotuneStatus(symbol = null) {
    let url = `${API_BASE}/autotune/status`
    if (symbol) url += `?symbol=${encodeURIComponent(symbol)}`
    return fetchJSON(url)
  },

  async healthCheck() {
    return fetchJSON(`${API_BASE}/healthz`)
  },
}

// ============================================================================
// AI/ML APIs
// ============================================================================

export const aiAPI = {
  // Chart & Technical Analysis
  async getChartOHLCV(ticker, timeframe = '1h', bars = 500) {
    return fetchJSON(`${API_BASE}/api/v1/ai/chart/ohlcv`, {
      method: 'POST',
      body: JSON.stringify({ ticker, timeframe, bars }),
    })
  },

  async getSupportResistance(ticker, timeframe = '1h', bars = 200) {
    return fetchJSON(
      `${API_BASE}/api/v1/ai/chart/support-resistance/${ticker}?timeframe=${timeframe}&bars=${bars}`
    )
  },

  // Features
  async generateFeatures(ticker, timeframe = '1h', bars = 500) {
    return fetchJSON(`${API_BASE}/api/v1/ai/features/generate`, {
      method: 'POST',
      body: JSON.stringify({ ticker, timeframe, bars }),
    })
  },

  // Models
  async trainModel(modelName, ticker, timeframe = '1h', bars = 1000, totalTimesteps = 50000) {
    return fetchJSON(`${API_BASE}/api/v1/ai/model/train`, {
      method: 'POST',
      body: JSON.stringify({
        model_name: modelName,
        ticker,
        timeframe,
        bars,
        total_timesteps: totalTimesteps,
      }),
    })
  },

  async getModelPrediction(ticker, modelName = 'trading_agent_v1') {
    return fetchJSON(
      `${API_BASE}/api/v1/ai/model/predict/${ticker}?model_name=${modelName}`
    )
  },

  // Signals
  async getAggregatedSignal(ticker) {
    return fetchJSON(`${API_BASE}/api/v1/ai/signal/aggregate/${ticker}`)
  },

  // TradingView Webhook
  async sendTradingViewWebhook(payload) {
    return fetchJSON(`${API_BASE}/api/v1/ai/tradingview/webhook`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },
}

// ============================================================================
// Autonomous Trading APIs
// ============================================================================

export const autonomousTradingAPI = {
  // Trading Settings
  async getSettings() {
    return fetchJSON(`${API_BASE}/api/v1/trading/settings`)
  },

  async updateSettings(settings) {
    return fetchJSON(`${API_BASE}/api/v1/trading/settings`, {
      method: 'PUT',
      body: JSON.stringify(settings),
    })
  },

  // Kill Switch
  async activateKillSwitch(reason = 'Manual activation') {
    return fetchJSON(`${API_BASE}/api/v1/trading/kill-switch/activate`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    })
  },

  async deactivateKillSwitch() {
    return fetchJSON(`${API_BASE}/api/v1/trading/kill-switch/deactivate`, {
      method: 'POST',
    })
  },

  async getKillSwitchStatus() {
    return fetchJSON(`${API_BASE}/api/v1/trading/kill-switch/status`)
  },

  // Pending Signals
  async getPendingSignals() {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/pending`)
  },

  async getSignalHistory(limit = 100) {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/history?limit=${limit}`)
  },

  async getSignal(signalId) {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/${signalId}`)
  },

  async approveSignal(signalId) {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/${signalId}/approve`, {
      method: 'POST',
    })
  },

  async rejectSignal(signalId, reason = 'Manual rejection') {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/${signalId}/reject?reason=${encodeURIComponent(reason)}`, {
      method: 'POST',
    })
  },

  // Manual Signal
  async submitManualSignal(symbol, action, confidence = 0.8, sizeUsd = null, reason = 'Manual signal') {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/manual`, {
      method: 'POST',
      body: JSON.stringify({
        symbol,
        action,
        confidence,
        size_usd: sizeUsd,
        reason,
      }),
    })
  },

  // Execution
  async executeSignal(signalId) {
    return fetchJSON(`${API_BASE}/api/v1/trading/execute/${signalId}`, {
      method: 'POST',
    })
  },

  // Status & Stats
  async getTradingStatus() {
    return fetchJSON(`${API_BASE}/api/v1/trading/status`)
  },

  async getDailyStats() {
    return fetchJSON(`${API_BASE}/api/v1/trading/stats/daily`)
  },

  async getActivityLog(limit = 50) {
    return fetchJSON(`${API_BASE}/api/v1/trading/activity?limit=${limit}`)
  },

  // Risk Check
  async checkRisk(symbol, sizeUsd = 100) {
    return fetchJSON(`${API_BASE}/api/v1/trading/risk-check/${symbol}?size_usd=${sizeUsd}`)
  },

  // Position Sizing
  async calculatePositionSize(confidence = 0.7, accountBalance = null) {
    let url = `${API_BASE}/api/v1/trading/position-size?confidence=${confidence}`
    if (accountBalance) url += `&account_balance=${accountBalance}`
    return fetchJSON(url)
  },

  // Autonomous Loop Control
  async startAutonomousLoop(intervalSeconds = 30) {
    return fetchJSON(`${API_BASE}/api/v1/trading/autonomous/start?interval_seconds=${intervalSeconds}`, {
      method: 'POST',
    })
  },

  async stopAutonomousLoop() {
    return fetchJSON(`${API_BASE}/api/v1/trading/autonomous/stop`, {
      method: 'POST',
    })
  },

  async getAutonomousStatus() {
    return fetchJSON(`${API_BASE}/api/v1/trading/autonomous/status`)
  },

  // Signal Simulation
  async simulateSignal(signalId) {
    return fetchJSON(`${API_BASE}/api/v1/trading/signals/${signalId}/simulate`, {
      method: 'POST',
    })
  },

  // Market Regime
  async getMarketRegime(ticker, timeframe = '1h') {
    return fetchJSON(`${API_BASE}/api/v1/trading/market-regime/${encodeURIComponent(ticker)}?timeframe=${timeframe}`)
  },

  async getMarketRegimeHistory(ticker, days = 30) {
    return fetchJSON(`${API_BASE}/api/v1/trading/market-regime/${encodeURIComponent(ticker)}/history?days=${days}`)
  },

  // Feature Store
  async getStoredFeatures(ticker, limit = 100) {
    return fetchJSON(`${API_BASE}/api/v1/trading/features/${encodeURIComponent(ticker)}?limit=${limit}`)
  },

  async getLatestFeatures(ticker) {
    return fetchJSON(`${API_BASE}/api/v1/trading/features/${encodeURIComponent(ticker)}/latest`)
  },

  async getFeatureStatistics() {
    return fetchJSON(`${API_BASE}/api/v1/trading/features/statistics`)
  },
}

// ============================================================================
// Analytics APIs
// ============================================================================

export const analyticsAPI = {
  // Trade History
  async getTradeHistory(params = {}) {
    const { limit = 100, offset = 0, symbol, status, side, startDate, endDate } = params
    const queryParams = new URLSearchParams({ limit, offset })
    if (symbol) queryParams.append('symbol', symbol)
    if (status) queryParams.append('status', status)
    if (side) queryParams.append('side', side)
    if (startDate) queryParams.append('start_date', startDate)
    if (endDate) queryParams.append('end_date', endDate)
    return fetchJSON(`${API_BASE}/api/v1/analytics/trades/history?${queryParams}`)
  },

  async getTradeMetrics(period = 'all', symbol = null) {
    let url = `${API_BASE}/api/v1/analytics/trades/metrics?period=${period}`
    if (symbol) url += `&symbol=${encodeURIComponent(symbol)}`
    return fetchJSON(url)
  },

  async getDailyPerformance(days = 30) {
    return fetchJSON(`${API_BASE}/api/v1/analytics/trades/daily-performance?days=${days}`)
  },

  async getTradesBySymbol(period = 'month') {
    return fetchJSON(`${API_BASE}/api/v1/analytics/trades/by-symbol?period=${period}`)
  },

  // Portfolio
  async getPortfolioPositions() {
    return fetchJSON(`${API_BASE}/api/v1/analytics/portfolio/positions`)
  },

  async getPortfolioSummary() {
    return fetchJSON(`${API_BASE}/api/v1/analytics/portfolio/summary`)
  },

  async getEquityCurve(days = 30) {
    return fetchJSON(`${API_BASE}/api/v1/analytics/portfolio/equity-curve?days=${days}`)
  },

  // Risk
  async getRiskMetrics() {
    return fetchJSON(`${API_BASE}/api/v1/analytics/risk/metrics`)
  },

  async getRiskExposure() {
    return fetchJSON(`${API_BASE}/api/v1/analytics/risk/exposure`)
  },

  // Strategies
  async listStrategies() {
    return fetchJSON(`${API_BASE}/api/v1/analytics/strategies/list`)
  },

  async getStrategyPerformance(strategyName) {
    return fetchJSON(`${API_BASE}/api/v1/analytics/strategies/${encodeURIComponent(strategyName)}/performance`)
  },
}

// ============================================================================
// Polling Utilities
// ============================================================================

export function createPoller(fetchFn, intervalMs = 10000) {
  let timeoutId = null
  let isRunning = false

  const poll = async (callback, errorCallback) => {
    if (!isRunning) return

    try {
      const data = await fetchFn()
      if (isRunning && callback) callback(data)
    } catch (error) {
      if (isRunning && errorCallback) errorCallback(error)
    }

    if (isRunning) {
      timeoutId = setTimeout(() => poll(callback, errorCallback), intervalMs)
    }
  }

  return {
    start(callback, errorCallback) {
      isRunning = true
      poll(callback, errorCallback)
    },
    stop() {
      isRunning = false
      if (timeoutId) clearTimeout(timeoutId)
    },
  }
}

export { APIError }
