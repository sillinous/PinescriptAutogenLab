import React, { useState } from 'react'
import { aiAPI } from '../../services/api'
import { Section } from '../common/Section'
import { LoadingSpinner } from '../common/Loading'

export function FeatureExplorer({ symbol = 'BTCUSDT' }) {
  const [features, setFeatures] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [timeframe, setTimeframe] = useState('1h')
  const [bars, setBars] = useState(500)

  const generateFeatures = async () => {
    setLoading(true)
    setError(null)

    try {
      const data = await aiAPI.generateFeatures(symbol, timeframe, bars)
      setFeatures(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Section title="🔬 ML Feature Explorer" badge="100+ Features">
      <div className="space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-[150px]">
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Timeframe
            </label>
            <select
              className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="1m">1 minute</option>
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
              <option value="30m">30 minutes</option>
              <option value="1h">1 hour</option>
              <option value="4h">4 hours</option>
              <option value="1d">1 day</option>
            </select>
          </div>

          <div className="flex-1 min-w-[150px]">
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Historical Bars
            </label>
            <input
              type="number"
              min="100"
              max="2000"
              className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
              value={bars}
              onChange={(e) => setBars(parseInt(e.target.value))}
            />
          </div>

          <button
            onClick={generateFeatures}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center gap-2 text-sm font-medium"
          >
            {loading ? (
              <>
                <LoadingSpinner size="sm" />
                Generating...
              </>
            ) : (
              <>
                🚀 Generate Features
              </>
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="text-xl">⚠️</div>
              <div>
                <div className="font-semibold text-red-900">Error</div>
                <div className="text-sm text-red-700 mt-1">{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {features && (
          <div className="space-y-4">
            {/* Summary */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-blue-900">
                  {features.num_features}
                </div>
                <div className="text-xs text-blue-700">Total Features</div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 border-2 border-purple-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-purple-900">{symbol}</div>
                <div className="text-xs text-purple-700">Ticker</div>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 border-2 border-green-200 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-green-900">{timeframe}</div>
                <div className="text-xs text-green-700">Timeframe</div>
              </div>
            </div>

            {/* Feature List */}
            <div className="bg-gray-50 rounded-lg p-4 border">
              <div className="text-xs font-medium text-gray-700 mb-3">
                Generated Features ({features.features?.length || 0})
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-96 overflow-y-auto">
                {features.features?.map((feature, idx) => (
                  <div
                    key={idx}
                    className="text-xs px-2 py-1.5 bg-white rounded border text-gray-700 font-mono truncate"
                    title={feature}
                  >
                    {feature}
                  </div>
                ))}
              </div>
            </div>

            {/* Sample Data */}
            {features.sample_data && features.sample_data.length > 0 && (
              <div>
                <div className="text-xs font-medium text-gray-700 mb-2">
                  Sample Values (Latest {features.sample_data.length} bars)
                </div>
                <div className="bg-gray-900 rounded-lg p-3 overflow-auto max-h-64">
                  <pre className="text-xs text-green-400 font-mono">
                    {JSON.stringify(features.sample_data, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Info */}
        {!features && !error && !loading && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm">
            <div className="font-semibold text-blue-900 mb-2">
              📊 About ML Features
            </div>
            <ul className="text-blue-700 space-y-1 text-xs">
              <li>• Generates 100+ technical, statistical, and pattern-based features</li>
              <li>• Features include: RSI, MACD, Bollinger Bands, moving averages, and more</li>
              <li>• Used for training machine learning models</li>
              <li>• Click "Generate Features" to see all available features for {symbol}</li>
            </ul>
          </div>
        )}
      </div>
    </Section>
  )
}
