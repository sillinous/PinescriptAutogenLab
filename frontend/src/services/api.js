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

  async getABStatus() {
    return fetchJSON(`${API_BASE}/ab/status`)
  },

  async getAutotuneStatus() {
    return fetchJSON(`${API_BASE}/autotune/status`)
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
