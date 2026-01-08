// frontend/src/components/trading/SignalQueueDashboard.jsx
/**
 * Signal Queue Dashboard Component
 *
 * Displays and manages pending trading signals with:
 * - Signal queue with approval/rejection workflow
 * - Signal simulation before execution
 * - Signal history and performance tracking
 * - Manual signal submission
 */

import React, { useState, useEffect, useCallback } from 'react'
import { autonomousTradingAPI } from '../../services/api'
import Card from '../common/Card'
import Loading from '../common/Loading'

// ============================================================================
// Sub-Components
// ============================================================================

function SignalBadge({ action }) {
  const colors = {
    buy: 'bg-green-600',
    sell: 'bg-red-600',
    hold: 'bg-gray-600',
  }
  return (
    <span className={`px-2 py-1 rounded text-xs font-bold text-white ${colors[action] || 'bg-gray-600'}`}>
      {action.toUpperCase()}
    </span>
  )
}

function ConfidenceMeter({ value }) {
  const pct = (value * 100).toFixed(0)
  const color = value >= 0.8 ? 'bg-green-500' : value >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-400">{pct}%</span>
    </div>
  )
}

function SimulationResult({ simulation }) {
  if (!simulation) return null

  const { estimated_pnl, risk_reward_ratio, win_probability, recommendation } = simulation
  const pnlColor = estimated_pnl >= 0 ? 'text-green-500' : 'text-red-500'

  return (
    <div className="bg-gray-900 rounded-lg p-4 mt-4 border border-gray-700">
      <h4 className="text-sm font-medium text-gray-300 mb-3">Simulation Results</h4>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <p className="text-xs text-gray-500">Est. P&L</p>
          <p className={`font-bold ${pnlColor}`}>${estimated_pnl?.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Risk/Reward</p>
          <p className="font-bold text-white">{risk_reward_ratio?.toFixed(2)}x</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Win Prob</p>
          <p className="font-bold text-white">{(win_probability * 100).toFixed(0)}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Recommendation</p>
          <span className={`px-2 py-1 rounded text-xs font-bold ${
            recommendation === 'execute' ? 'bg-green-600 text-white' :
            recommendation === 'modify' ? 'bg-yellow-600 text-white' :
            'bg-red-600 text-white'
          }`}>
            {recommendation?.toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  )
}

function SignalCard({ signal, onApprove, onReject, onSimulate }) {
  const [simulating, setSimulating] = useState(false)
  const [simulation, setSimulation] = useState(null)
  const [processing, setProcessing] = useState(false)

  const handleSimulate = async () => {
    setSimulating(true)
    try {
      const result = await onSimulate(signal.id)
      setSimulation(result)
    } finally {
      setSimulating(false)
    }
  }

  const handleApprove = async () => {
    setProcessing(true)
    try {
      await onApprove(signal.id)
    } finally {
      setProcessing(false)
    }
  }

  const handleReject = async () => {
    const reason = prompt('Reason for rejection:', 'Manual rejection')
    if (reason === null) return

    setProcessing(true)
    try {
      await onReject(signal.id, reason)
    } finally {
      setProcessing(false)
    }
  }

  const timeAgo = (timestamp) => {
    const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000)
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
    return `${Math.floor(seconds / 86400)}d ago`
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex justify-between items-start mb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg font-bold text-white">{signal.symbol}</span>
            <SignalBadge action={signal.action} />
          </div>
          <p className="text-xs text-gray-500">
            {signal.source || 'AI'} • {timeAgo(signal.created_at)}
          </p>
        </div>
        <ConfidenceMeter value={signal.confidence} />
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm mb-4">
        <div>
          <p className="text-gray-500">Size</p>
          <p className="text-white">${signal.size_usd?.toFixed(2) || '-'}</p>
        </div>
        <div>
          <p className="text-gray-500">Strategy</p>
          <p className="text-white">{signal.strategy_name || 'Default'}</p>
        </div>
        {signal.reason && (
          <div className="col-span-2">
            <p className="text-gray-500">Reason</p>
            <p className="text-gray-300 text-xs">{signal.reason}</p>
          </div>
        )}
      </div>

      <SimulationResult simulation={simulation} />

      <div className="flex gap-2 mt-4">
        <button
          onClick={handleSimulate}
          disabled={simulating}
          className="flex-1 px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition"
        >
          {simulating ? 'Simulating...' : 'Simulate'}
        </button>
        <button
          onClick={handleApprove}
          disabled={processing}
          className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition"
        >
          {processing ? '...' : 'Approve'}
        </button>
        <button
          onClick={handleReject}
          disabled={processing}
          className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm transition"
        >
          {processing ? '...' : 'Reject'}
        </button>
      </div>
    </div>
  )
}

function HistoryRow({ signal }) {
  const statusColors = {
    approved: 'text-green-500',
    rejected: 'text-red-500',
    executed: 'text-blue-500',
    expired: 'text-gray-500',
  }

  return (
    <tr className="border-b border-gray-700">
      <td className="py-2 px-3 text-white">{signal.symbol}</td>
      <td className="py-2 px-3">
        <SignalBadge action={signal.action} />
      </td>
      <td className="py-2 px-3 text-gray-300">{(signal.confidence * 100).toFixed(0)}%</td>
      <td className="py-2 px-3 text-gray-300">${signal.size_usd?.toFixed(2) || '-'}</td>
      <td className={`py-2 px-3 ${statusColors[signal.status] || 'text-gray-300'}`}>
        {signal.status}
      </td>
      <td className="py-2 px-3 text-gray-500 text-xs">
        {new Date(signal.created_at).toLocaleString()}
      </td>
    </tr>
  )
}

function ManualSignalForm({ onSubmit }) {
  const [symbol, setSymbol] = useState('BTC_USDT')
  const [action, setAction] = useState('buy')
  const [confidence, setConfidence] = useState(0.8)
  const [sizeUsd, setSizeUsd] = useState(100)
  const [reason, setReason] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await onSubmit(symbol, action, confidence, sizeUsd, reason || 'Manual signal')
      setReason('')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-2 md:grid-cols-6 gap-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Symbol</label>
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Action</label>
        <select
          value={action}
          onChange={(e) => setAction(e.target.value)}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
        >
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
          <option value="hold">Hold</option>
        </select>
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Confidence</label>
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          value={confidence}
          onChange={(e) => setConfidence(parseFloat(e.target.value))}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Size (USD)</label>
        <input
          type="number"
          min="0"
          value={sizeUsd}
          onChange={(e) => setSizeUsd(parseFloat(e.target.value))}
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Reason</label>
        <input
          type="text"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="Optional"
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
        />
      </div>
      <div className="flex items-end">
        <button
          type="submit"
          disabled={submitting}
          className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition"
        >
          {submitting ? 'Submitting...' : 'Submit Signal'}
        </button>
      </div>
    </form>
  )
}

// ============================================================================
// Main Component
// ============================================================================

export default function SignalQueueDashboard() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [pendingSignals, setPendingSignals] = useState([])
  const [signalHistory, setSignalHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      const [pending, history] = await Promise.allSettled([
        autonomousTradingAPI.getPendingSignals(),
        autonomousTradingAPI.getSignalHistory(50)
      ])

      if (pending.status === 'fulfilled') setPendingSignals(pending.value.signals || [])
      if (history.status === 'fulfilled') setSignalHistory(history.value.signals || [])
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000) // Poll every 10s
    return () => clearInterval(interval)
  }, [fetchData])

  const handleApprove = async (signalId) => {
    await autonomousTradingAPI.approveSignal(signalId)
    fetchData()
  }

  const handleReject = async (signalId, reason) => {
    await autonomousTradingAPI.rejectSignal(signalId, reason)
    fetchData()
  }

  const handleSimulate = async (signalId) => {
    return await autonomousTradingAPI.simulateSignal(signalId)
  }

  const handleManualSignal = async (symbol, action, confidence, sizeUsd, reason) => {
    await autonomousTradingAPI.submitManualSignal(symbol, action, confidence, sizeUsd, reason)
    fetchData()
  }

  if (loading) {
    return <Loading message="Loading signal queue..." />
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white">Signal Queue</h2>
          <p className="text-gray-400 text-sm">
            {pendingSignals.length} pending signals awaiting review
          </p>
        </div>
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition"
        >
          {showHistory ? 'Hide History' : 'Show History'}
        </button>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Manual Signal Form */}
      <Card title="Submit Manual Signal" subtitle="Create a new trading signal manually">
        <ManualSignalForm onSubmit={handleManualSignal} />
      </Card>

      {/* Pending Signals */}
      <div>
        <h3 className="text-lg font-medium text-white mb-4">Pending Signals</h3>
        {pendingSignals.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {pendingSignals.map((signal) => (
              <SignalCard
                key={signal.id}
                signal={signal}
                onApprove={handleApprove}
                onReject={handleReject}
                onSimulate={handleSimulate}
              />
            ))}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500">
            No pending signals. New signals will appear here automatically.
          </div>
        )}
      </div>

      {/* Signal History */}
      {showHistory && (
        <Card title="Signal History" subtitle="Recent signal activity">
          {signalHistory.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                    <th className="py-2 px-3">Symbol</th>
                    <th className="py-2 px-3">Action</th>
                    <th className="py-2 px-3">Confidence</th>
                    <th className="py-2 px-3">Size</th>
                    <th className="py-2 px-3">Status</th>
                    <th className="py-2 px-3">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {signalHistory.map((signal, idx) => (
                    <HistoryRow key={signal.id || idx} signal={signal} />
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No signal history available</p>
          )}
        </Card>
      )}

      {/* Refresh Button */}
      <div className="flex justify-center">
        <button
          onClick={fetchData}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition"
        >
          Refresh Queue
        </button>
      </div>
    </div>
  )
}
