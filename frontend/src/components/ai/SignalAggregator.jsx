import React, { useEffect, useState } from 'react'
import { aiAPI } from '../../services/api'
import { Section } from '../common/Section'
import { Card } from '../common/Card'
import { LoadingSpinner } from '../common/Loading'

export function SignalAggregator({ symbol, autoRefresh = true }) {
  const [signal, setSignal] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchSignal = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await aiAPI.getAggregatedSignal(symbol)
      setSignal(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSignal()

    if (autoRefresh) {
      const interval = setInterval(fetchSignal, 15000) // Refresh every 15s
      return () => clearInterval(interval)
    }
  }, [symbol, autoRefresh])

  if (loading && !signal) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-50 border-2 border-gray-200 rounded-xl p-6 text-center">
        <div className="text-4xl mb-2">⚠️</div>
        <div className="text-sm text-gray-600">{error}</div>
      </div>
    )
  }

  if (!signal) return null

  // Safe defaults for signal data
  const action = (signal.action || 'HOLD').toLowerCase()
  const confidence = signal.confidence || 0
  const consensus = signal.consensus || 0
  const positionSize = signal.recommended_position_size || 0
  const timestamp = signal.timestamp || new Date().toISOString()
  const sources = signal.sources || []

  const actionStatus = {
    buy: 'success',
    sell: 'error',
    hold: 'warning',
  }

  const actionIcons = {
    buy: '🚀',
    sell: '📉',
    hold: '⏸️',
  }

  return (
    <div className="bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-200 rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-bold text-gray-800">🎯 Aggregated Trading Signal</h2>
          {loading && <LoadingSpinner size="sm" />}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {new Date(timestamp).toLocaleTimeString()}
          </span>
          <button
            onClick={fetchSignal}
            className="text-xs px-2 py-1 bg-white rounded-full hover:bg-gray-100 transition-colors"
          >
            🔄
          </button>
        </div>
      </div>

      {/* Main Signal */}
      <div className="grid md:grid-cols-4 gap-4 mb-4">
        <Card
          title="Recommended Action"
          value={
            <div className="flex items-center gap-2">
              <span>{actionIcons[action]}</span>
              <span>{action.toUpperCase()}</span>
            </div>
          }
          status={actionStatus[action]}
        />
        <Card
          title="Signal Confidence"
          value={`${(confidence * 100).toFixed(1)}%`}
          subtitle={confidence > 0.7 ? 'High Confidence' : 'Moderate'}
          status={confidence > 0.7 ? 'success' : 'warning'}
        />
        <Card
          title="Consensus Score"
          value={`${(consensus * 100).toFixed(1)}%`}
          subtitle="Source Agreement"
        />
        <Card
          title="Position Size"
          value={`${(positionSize * 100).toFixed(0)}%`}
          subtitle="Recommended"
        />
      </div>

      {/* Signal Sources */}
      {sources && sources.length > 0 && (
        <div className="bg-white/70 backdrop-blur rounded-lg p-4">
          <div className="text-xs font-medium text-gray-700 mb-2">
            Contributing Signals ({sources.length})
          </div>
          <div className="flex flex-wrap gap-2">
            {sources.map((source, idx) => (
              <span
                key={idx}
                className="text-xs px-3 py-1.5 bg-white rounded-full border border-purple-200 shadow-sm"
              >
                {source}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Demo Mode Warning */}
      {signal.message && (
        <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <div className="text-xs text-yellow-800">
            ℹ️ {signal.message}
          </div>
        </div>
      )}

      {/* Action Recommendation */}
      <div className="mt-4 bg-white/50 rounded-lg p-3 border border-purple-200">
        <div className="text-sm">
          <span className="font-semibold">Recommendation:</span>{' '}
          {action === 'buy' && (
            <span className="text-green-700">
              Consider opening a long position with {(positionSize * 100).toFixed(0)}% of available capital
            </span>
          )}
          {action === 'sell' && (
            <span className="text-red-700">
              Consider closing long positions or opening short with {(positionSize * 100).toFixed(0)}% size
            </span>
          )}
          {action === 'hold' && (
            <span className="text-yellow-700">
              No strong signal detected. Maintain current positions or wait for clearer opportunity
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
