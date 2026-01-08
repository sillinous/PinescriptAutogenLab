// frontend/src/components/trading/MarketRegimeDashboard.jsx
/**
 * Market Regime Dashboard Component
 *
 * Displays market regime detection and analysis:
 * - Current regime for selected tickers
 * - Regime history visualization
 * - Feature store statistics
 * - Multi-ticker regime comparison
 */

import React, { useState, useEffect, useCallback } from 'react'
import { autonomousTradingAPI } from '../../services/api'
import Card from '../common/Card'
import Loading from '../common/Loading'

// ============================================================================
// Sub-Components
// ============================================================================

function RegimeBadge({ regime }) {
  const configs = {
    bullish_trending: { bg: 'bg-green-600', text: 'Bullish Trending', icon: '📈' },
    bearish_trending: { bg: 'bg-red-600', text: 'Bearish Trending', icon: '📉' },
    range_bound: { bg: 'bg-yellow-600', text: 'Range Bound', icon: '↔️' },
    high_volatility: { bg: 'bg-purple-600', text: 'High Volatility', icon: '⚡' },
    unknown: { bg: 'bg-gray-600', text: 'Unknown', icon: '❓' },
  }

  const config = configs[regime] || configs.unknown

  return (
    <span className={`px-3 py-1 rounded-lg text-sm font-medium text-white ${config.bg} inline-flex items-center gap-1`}>
      {config.icon} {config.text}
    </span>
  )
}

function ConfidenceBar({ value, label }) {
  const pct = (value * 100).toFixed(0)
  const color = value >= 0.7 ? 'bg-green-500' : value >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function RegimeCard({ ticker, data, onRefresh }) {
  if (!data) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-bold text-white mb-2">{ticker}</h3>
        <p className="text-gray-500">Loading...</p>
      </div>
    )
  }

  const { regime, confidence, trend_strength, volatility_percentile, sma_20, sma_50, current_price, atr } = data

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-bold text-white">{ticker}</h3>
          <p className="text-gray-400 text-sm">${current_price?.toFixed(2)}</p>
        </div>
        <RegimeBadge regime={regime} />
      </div>

      <div className="space-y-1 mb-4">
        <ConfidenceBar value={confidence || 0} label="Confidence" />
        <ConfidenceBar value={trend_strength || 0} label="Trend Strength" />
        <ConfidenceBar value={volatility_percentile || 0} label="Volatility" />
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-500">SMA 20</p>
          <p className="text-white">${sma_20?.toFixed(2) || '-'}</p>
        </div>
        <div>
          <p className="text-gray-500">SMA 50</p>
          <p className="text-white">${sma_50?.toFixed(2) || '-'}</p>
        </div>
        <div>
          <p className="text-gray-500">ATR</p>
          <p className="text-white">${atr?.toFixed(2) || '-'}</p>
        </div>
        <div>
          <p className="text-gray-500">Timeframe</p>
          <p className="text-white">{data.timeframe || '1h'}</p>
        </div>
      </div>

      <button
        onClick={() => onRefresh(ticker)}
        className="w-full mt-4 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition"
      >
        Refresh
      </button>
    </div>
  )
}

function RegimeHistoryChart({ history }) {
  if (!history || history.length === 0) {
    return <p className="text-gray-500 text-center py-8">No regime history available</p>
  }

  const regimeColors = {
    bullish_trending: '#22c55e',
    bearish_trending: '#ef4444',
    range_bound: '#eab308',
    high_volatility: '#a855f7',
  }

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-1 min-w-max">
        {history.map((entry, idx) => (
          <div
            key={idx}
            className="w-4 h-16 rounded-sm cursor-pointer transition hover:opacity-80"
            style={{ backgroundColor: regimeColors[entry.regime] || '#6b7280' }}
            title={`${entry.date}: ${entry.regime}`}
          />
        ))}
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-2">
        <span>{history[0]?.date}</span>
        <span>{history[history.length - 1]?.date}</span>
      </div>
    </div>
  )
}

function FeatureStoreStats({ stats }) {
  if (!stats) return null

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <p className="text-gray-400 text-sm">Total Features</p>
        <p className="text-2xl font-bold text-white">{stats.total_records?.toLocaleString() || 0}</p>
      </div>
      <div className="bg-gray-800 rounded-lg p-4">
        <p className="text-gray-400 text-sm">Unique Tickers</p>
        <p className="text-2xl font-bold text-white">{stats.unique_tickers || 0}</p>
      </div>
      <div className="bg-gray-800 rounded-lg p-4">
        <p className="text-gray-400 text-sm">Latest Update</p>
        <p className="text-lg font-bold text-white">
          {stats.latest_update ? new Date(stats.latest_update).toLocaleString() : '-'}
        </p>
      </div>
      <div className="bg-gray-800 rounded-lg p-4">
        <p className="text-gray-400 text-sm">Feature Count</p>
        <p className="text-2xl font-bold text-white">{stats.feature_count || '-'}</p>
      </div>
    </div>
  )
}

// ============================================================================
// Main Component
// ============================================================================

const DEFAULT_TICKERS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT']

export default function MarketRegimeDashboard() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [regimeData, setRegimeData] = useState({})
  const [selectedTicker, setSelectedTicker] = useState('BTC_USDT')
  const [regimeHistory, setRegimeHistory] = useState([])
  const [featureStats, setFeatureStats] = useState(null)
  const [tickers, setTickers] = useState(DEFAULT_TICKERS)
  const [newTicker, setNewTicker] = useState('')
  const [timeframe, setTimeframe] = useState('1h')

  const fetchRegimeForTicker = useCallback(async (ticker) => {
    try {
      const data = await autonomousTradingAPI.getMarketRegime(ticker, timeframe)
      setRegimeData(prev => ({ ...prev, [ticker]: data }))
    } catch (err) {
      console.error(`Failed to fetch regime for ${ticker}:`, err)
    }
  }, [timeframe])

  const fetchAllData = useCallback(async () => {
    setLoading(true)
    try {
      // Fetch regime for all tickers in parallel
      await Promise.all(tickers.map(fetchRegimeForTicker))

      // Fetch regime history for selected ticker
      const history = await autonomousTradingAPI.getMarketRegimeHistory(selectedTicker, 30)
      setRegimeHistory(history.history || [])

      // Fetch feature store stats
      const stats = await autonomousTradingAPI.getFeatureStatistics()
      setFeatureStats(stats)

      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [tickers, selectedTicker, fetchRegimeForTicker])

  useEffect(() => {
    fetchAllData()
    const interval = setInterval(fetchAllData, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [fetchAllData])

  const handleAddTicker = () => {
    const ticker = newTicker.trim().toUpperCase()
    if (ticker && !tickers.includes(ticker)) {
      setTickers(prev => [...prev, ticker])
      setNewTicker('')
      fetchRegimeForTicker(ticker)
    }
  }

  const handleRemoveTicker = (ticker) => {
    if (tickers.length > 1) {
      setTickers(prev => prev.filter(t => t !== ticker))
      if (selectedTicker === ticker) {
        setSelectedTicker(tickers.find(t => t !== ticker) || tickers[0])
      }
    }
  }

  if (loading && Object.keys(regimeData).length === 0) {
    return <Loading message="Loading market regime data..." />
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white">Market Regime Detection</h2>
          <p className="text-gray-400 text-sm">
            Real-time market regime analysis using technical indicators
          </p>
        </div>
        <div className="flex gap-2">
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
          </select>
          <button
            onClick={fetchAllData}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition"
          >
            Refresh All
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Add Ticker */}
      <Card title="Tracked Tickers">
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value)}
            placeholder="e.g., DOGE_USDT"
            className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            onKeyDown={(e) => e.key === 'Enter' && handleAddTicker()}
          />
          <button
            onClick={handleAddTicker}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition"
          >
            Add Ticker
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {tickers.map(ticker => (
            <div
              key={ticker}
              className={`px-3 py-1 rounded-lg text-sm flex items-center gap-2 cursor-pointer transition ${
                selectedTicker === ticker
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              onClick={() => setSelectedTicker(ticker)}
            >
              {ticker}
              {tickers.length > 1 && (
                <button
                  onClick={(e) => { e.stopPropagation(); handleRemoveTicker(ticker); }}
                  className="text-gray-400 hover:text-red-400"
                >
                  x
                </button>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Regime Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {tickers.map(ticker => (
          <RegimeCard
            key={ticker}
            ticker={ticker}
            data={regimeData[ticker]}
            onRefresh={fetchRegimeForTicker}
          />
        ))}
      </div>

      {/* Regime History */}
      <Card
        title={`${selectedTicker} Regime History`}
        subtitle="Last 30 days of regime changes"
      >
        <RegimeHistoryChart history={regimeHistory} />
        <div className="flex justify-center gap-4 mt-4 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-green-500"></span> Bullish
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-500"></span> Bearish
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-yellow-500"></span> Range
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-purple-500"></span> High Vol
          </span>
        </div>
      </Card>

      {/* Feature Store Stats */}
      <Card title="Feature Store Statistics" subtitle="ML feature storage overview">
        <FeatureStoreStats stats={featureStats} />
      </Card>
    </div>
  )
}
