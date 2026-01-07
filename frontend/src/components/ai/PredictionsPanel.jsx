import React, { useEffect, useState } from 'react'
import { aiAPI } from '../../services/api'
import { Section } from '../common/Section'
import { MetricCard } from '../common/Card'
import { LoadingCard } from '../common/Loading'

export function PredictionsPanel({ symbol, modelName = 'trading_agent_v1', autoRefresh = true }) {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchPrediction = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await aiAPI.getModelPrediction(symbol, modelName)
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPrediction()

    if (autoRefresh) {
      const interval = setInterval(fetchPrediction, 30000) // Refresh every 30s
      return () => clearInterval(interval)
    }
  }, [symbol, modelName, autoRefresh])

  if (loading && !prediction) {
    return <LoadingCard message="Loading AI predictions..." />
  }

  // Example prediction data when no model is available
  const examplePrediction = {
    action: 'HOLD',
    confidence: 0.72,
    all_action_probs: {
      BUY: 0.18,
      HOLD: 0.72,
      SELL: 0.10,
    },
    market_context: {
      current_price: 94250.00,
      ticker: symbol,
      timeframe: '1h',
    },
    model_name: 'Example Model',
    timestamp: new Date().toISOString(),
  }

  // Show example prediction when no model is available
  if (error) {
    const exAction = examplePrediction.action
    const exActionColors = { BUY: 'green', SELL: 'red', HOLD: 'yellow' }
    const exActionIcons = { BUY: '🚀', SELL: '📉', HOLD: '⏸️' }
    const exColor = exActionColors[exAction]
    const exIcon = exActionIcons[exAction]

    return (
      <Section
        title="🤖 AI Trading Predictions"
        badge="Example"
        action={
          <span className="text-xs px-3 py-1 bg-purple-100 text-purple-700 rounded-full">
            Demo Mode
          </span>
        }
      >
        <div className="space-y-4">
          {/* Example Notice */}
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <span className="text-lg">💡</span>
              <div className="text-sm">
                <span className="font-semibold text-purple-900">Example Prediction</span>
                <p className="text-purple-700 mt-1">
                  This demonstrates how AI predictions appear. Train your own model using the
                  <strong> Deep Learning</strong> tab to get real predictions based on market data.
                </p>
              </div>
            </div>
          </div>

          {/* Example Prediction Display */}
          <div className={`bg-gradient-to-br from-${exColor}-50 to-${exColor}-100 border-2 border-${exColor}-300 rounded-xl p-6 opacity-90`}>
            <div className="text-center">
              <div className="text-5xl mb-2">{exIcon}</div>
              <div className={`text-3xl font-bold text-${exColor}-900 mb-1`}>
                {exAction}
              </div>
              <div className={`text-lg text-${exColor}-700`}>
                {(examplePrediction.confidence * 100).toFixed(1)}% Confidence
              </div>
            </div>
          </div>

          {/* Example Probability Breakdown */}
          <div>
            <div className="text-xs font-medium text-gray-700 mb-2">
              Action Probabilities (Example)
            </div>
            <div className="space-y-2">
              {Object.entries(examplePrediction.all_action_probs)
                .sort((a, b) => b[1] - a[1])
                .map(([action, prob]) => (
                  <div key={action} className="flex items-center gap-3">
                    <div className="w-16 text-sm font-medium">{action}</div>
                    <div className="flex-1">
                      <div className="w-full h-6 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-6 ${
                            action === 'BUY'
                              ? 'bg-green-500'
                              : action === 'SELL'
                              ? 'bg-red-500'
                              : 'bg-yellow-500'
                          } flex items-center justify-end pr-2 transition-all duration-500`}
                          style={{ width: `${prob * 100}%` }}
                        >
                          <span className="text-xs font-semibold text-white">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Training CTA */}
          <div className="bg-gray-50 rounded-lg p-4 text-center">
            <p className="text-sm text-gray-600 mb-2">
              Ready to get real predictions?
            </p>
            <p className="text-xs text-gray-500">
              Navigate to <strong>Deep Learning</strong> → Train an LSTM or Transformer model →
              Predictions will update automatically
            </p>
          </div>
        </div>
      </Section>
    )
  }

  if (!prediction) return null

  const actionColors = {
    BUY: 'green',
    SELL: 'red',
    HOLD: 'yellow',
  }

  const actionIcons = {
    BUY: '🚀',
    SELL: '📉',
    HOLD: '⏸️',
  }

  const actionColor = actionColors[prediction.action] || 'blue'
  const actionIcon = actionIcons[prediction.action] || '🤖'

  return (
    <Section
      title="🤖 AI Trading Predictions"
      badge={prediction.model_name}
      action={
        <button
          onClick={fetchPrediction}
          className="text-xs px-3 py-1 bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
        >
          Refresh
        </button>
      }
    >
      <div className="space-y-4">
        {/* Main Prediction */}
        <div className={`bg-gradient-to-br from-${actionColor}-50 to-${actionColor}-100 border-2 border-${actionColor}-300 rounded-xl p-6`}>
          <div className="text-center">
            <div className="text-5xl mb-2">{actionIcon}</div>
            <div className={`text-3xl font-bold text-${actionColor}-900 mb-1`}>
              {prediction.action}
            </div>
            <div className={`text-lg text-${actionColor}-700`}>
              {(prediction.confidence * 100).toFixed(1)}% Confidence
            </div>
          </div>
        </div>

        {/* Probability Breakdown */}
        {prediction.all_action_probs && (
          <div>
            <div className="text-xs font-medium text-gray-700 mb-2">
              Action Probabilities
            </div>
            <div className="space-y-2">
              {Object.entries(prediction.all_action_probs)
                .sort((a, b) => b[1] - a[1])
                .map(([action, prob]) => (
                  <div key={action} className="flex items-center gap-3">
                    <div className="w-16 text-sm font-medium">{action}</div>
                    <div className="flex-1">
                      <div className="w-full h-6 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-6 ${
                            action === 'BUY'
                              ? 'bg-green-500'
                              : action === 'SELL'
                              ? 'bg-red-500'
                              : 'bg-yellow-500'
                          } flex items-center justify-end pr-2 transition-all duration-500`}
                          style={{ width: `${prob * 100}%` }}
                        >
                          <span className="text-xs font-semibold text-white">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Market Context */}
        {prediction.market_context && (
          <div className="grid grid-cols-3 gap-3">
            <MetricCard
              icon="💰"
              label="Current Price"
              value={`$${prediction.market_context.current_price?.toFixed(4) || 'N/A'}`}
              color="blue"
            />
            <MetricCard
              icon="📊"
              label="Ticker"
              value={prediction.market_context.ticker || symbol}
              color="purple"
            />
            <MetricCard
              icon="⏱️"
              label="Timeframe"
              value={prediction.market_context.timeframe || '1h'}
              color="blue"
            />
          </div>
        )}

        {/* Timestamp */}
        {prediction.timestamp && (
          <div className="text-xs text-gray-500 text-center">
            Last updated: {new Date(prediction.timestamp).toLocaleString()}
          </div>
        )}
      </div>
    </Section>
  )
}
