import React, { useState } from 'react'
import { aiAPI } from '../../services/api'
import { Section } from '../common/Section'
import { Card } from '../common/Card'
import { LoadingSpinner } from '../common/Loading'

export function ModelManagement({ symbol = 'BTCUSDT' }) {
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState(null)
  const [error, setError] = useState(null)

  const [formData, setFormData] = useState({
    modelName: '',
    ticker: symbol,
    timeframe: '1h',
    bars: 2000,
    totalTimesteps: 100000,
  })

  const handleTrain = async (e) => {
    e.preventDefault()
    setTraining(true)
    setError(null)
    setTrainResult(null)

    try {
      const result = await aiAPI.trainModel(
        formData.modelName || `model_${Date.now()}`,
        formData.ticker,
        formData.timeframe,
        parseInt(formData.bars),
        parseInt(formData.totalTimesteps)
      )
      setTrainResult(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setTraining(false)
    }
  }

  const presets = [
    { name: 'Quick Test', timesteps: 50000, bars: 1000, desc: '5-10 min' },
    { name: 'Standard', timesteps: 100000, bars: 2000, desc: '10-20 min' },
    { name: 'Production', timesteps: 500000, bars: 5000, desc: '30-60 min' },
  ]

  return (
    <Section title="🎓 AI Model Training Lab" badge="RL Agent">
      <div className="space-y-4">
        {/* Training Status */}
        {trainResult && (
          <div className="bg-green-50 border-2 border-green-200 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <div className="text-2xl">✅</div>
              <div className="flex-1">
                <div className="font-semibold text-green-900">Training Started</div>
                <div className="text-sm text-green-700 mt-1">
                  Model: <span className="font-mono">{trainResult.model_name}</span>
                </div>
                <div className="text-sm text-green-700">
                  Timesteps: {trainResult.total_timesteps.toLocaleString()}
                </div>
                <div className="text-xs text-green-600 mt-2">
                  {trainResult.message}
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <div className="text-2xl">⚠️</div>
              <div className="flex-1">
                <div className="font-semibold text-red-900">Training Failed</div>
                <div className="text-sm text-red-700 mt-1">{error}</div>
              </div>
            </div>
          </div>
        )}

        {/* Training Form */}
        <form onSubmit={handleTrain} className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Model Name
              </label>
              <input
                type="text"
                placeholder="e.g., btc_trader_v1"
                className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={formData.modelName}
                onChange={(e) => setFormData({ ...formData, modelName: e.target.value })}
              />
              <div className="text-xs text-gray-500 mt-1">
                Leave empty for auto-generated name
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Ticker
              </label>
              <input
                type="text"
                className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={formData.ticker}
                onChange={(e) => setFormData({ ...formData, ticker: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Timeframe
              </label>
              <select
                className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={formData.timeframe}
                onChange={(e) => setFormData({ ...formData, timeframe: e.target.value })}
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

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Historical Bars
              </label>
              <input
                type="number"
                min="100"
                max="10000"
                className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={formData.bars}
                onChange={(e) => setFormData({ ...formData, bars: e.target.value })}
                required
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Training Timesteps
              </label>
              <input
                type="number"
                min="10000"
                max="1000000"
                step="10000"
                className="w-full border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                value={formData.totalTimesteps}
                onChange={(e) =>
                  setFormData({ ...formData, totalTimesteps: e.target.value })
                }
                required
              />
              <div className="text-xs text-gray-500 mt-1">
                More timesteps = better model, but longer training time
              </div>
            </div>
          </div>

          {/* Presets */}
          <div>
            <div className="text-xs font-medium text-gray-700 mb-2">Quick Presets</div>
            <div className="grid grid-cols-3 gap-2">
              {presets.map((preset) => (
                <button
                  key={preset.name}
                  type="button"
                  className="border rounded-lg p-2 hover:bg-gray-50 transition-colors text-sm"
                  onClick={() =>
                    setFormData({
                      ...formData,
                      totalTimesteps: preset.timesteps,
                      bars: preset.bars,
                    })
                  }
                >
                  <div className="font-semibold">{preset.name}</div>
                  <div className="text-xs text-gray-500">{preset.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={training}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg px-4 py-3 font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {training ? (
              <>
                <LoadingSpinner size="sm" className="text-white" />
                Training in Background...
              </>
            ) : (
              <>
                🚀 Start Training
              </>
            )}
          </button>
        </form>

        {/* Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm">
          <div className="font-semibold text-blue-900 mb-1">Training Info</div>
          <ul className="text-blue-700 space-y-1 text-xs">
            <li>• Training runs in the background on the server</li>
            <li>• Model will be saved automatically when training completes</li>
            <li>• Check server logs for training progress</li>
            <li>• Recommended: 100K+ timesteps for production models</li>
          </ul>
        </div>
      </div>
    </Section>
  )
}
