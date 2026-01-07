import React, { useState, useEffect, useCallback } from 'react'
import { autonomousTradingAPI } from '../../services/api'
import { Section } from '../common/Section'
import { Card, MetricCard } from '../common/Card'
import { LoadingCard } from '../common/Loading'

// Trading mode options
const TRADING_MODES = [
  { value: 'off', label: 'Off', description: 'No trading' },
  { value: 'manual', label: 'Manual', description: 'All signals require approval' },
  { value: 'semi_auto', label: 'Semi-Auto', description: 'High-confidence auto, rest manual' },
  { value: 'full_auto', label: 'Full Auto', description: 'Fully autonomous' },
]

const RISK_PROFILES = [
  { value: 'conservative', label: 'Conservative', description: 'Lower risk, smaller positions' },
  { value: 'moderate', label: 'Moderate', description: 'Balanced approach' },
  { value: 'aggressive', label: 'Aggressive', description: 'Higher risk, larger positions' },
]

// Kill Switch Component
function KillSwitch({ active, reason, onActivate, onDeactivate }) {
  const [activating, setActivating] = useState(false)

  const handleToggle = async () => {
    setActivating(true)
    try {
      if (active) {
        await onDeactivate()
      } else {
        const reason = prompt('Enter reason for activating kill switch:', 'Manual safety stop')
        if (reason) {
          await onActivate(reason)
        }
      }
    } finally {
      setActivating(false)
    }
  }

  return (
    <div className={`p-4 rounded-xl border-2 ${active ? 'bg-red-50 border-red-500' : 'bg-green-50 border-green-500'}`}>
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-bold text-lg flex items-center gap-2">
            {active ? '🛑' : '✅'} Kill Switch
          </h3>
          <p className="text-sm text-gray-600">
            {active ? `ACTIVE: ${reason}` : 'Trading is enabled'}
          </p>
        </div>
        <button
          onClick={handleToggle}
          disabled={activating}
          className={`px-6 py-3 rounded-lg font-bold text-white transition-all ${
            active
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-red-600 hover:bg-red-700'
          } ${activating ? 'opacity-50' : ''}`}
        >
          {activating ? '...' : active ? 'Resume Trading' : 'STOP ALL TRADING'}
        </button>
      </div>
    </div>
  )
}

// Trading Status Overview
function TradingStatus({ status, dailyStats }) {
  if (!status) return <LoadingCard message="Loading status..." />

  const winRate = dailyStats?.win_rate || 0

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard
        icon={status.trading_enabled ? '🟢' : '🔴'}
        label="Trading"
        value={status.trading_mode.replace('_', ' ').toUpperCase()}
        color={status.trading_enabled ? 'green' : 'red'}
      />
      <MetricCard
        icon="📊"
        label="Today's Trades"
        value={`${status.today_trades} / ${status.today_trades + status.daily_limit_remaining}`}
        color="blue"
      />
      <MetricCard
        icon={status.today_pnl >= 0 ? '📈' : '📉'}
        label="Today's P&L"
        value={`$${status.today_pnl.toFixed(2)}`}
        color={status.today_pnl >= 0 ? 'green' : 'red'}
      />
      <MetricCard
        icon="🎯"
        label="Win Rate"
        value={`${winRate.toFixed(1)}%`}
        color={winRate >= 50 ? 'green' : 'yellow'}
      />
    </div>
  )
}

// Pending Signals Queue
function PendingSignalsQueue({ signals, onApprove, onReject, loading }) {
  if (loading) return <LoadingCard message="Loading signals..." />

  if (!signals || signals.length === 0) {
    return (
      <Section title="Pending Signals" badge="Queue">
        <div className="text-center text-gray-500 py-8">
          <span className="text-4xl">📭</span>
          <p className="mt-2">No pending signals</p>
          <p className="text-sm">Signals will appear here when generated</p>
        </div>
      </Section>
    )
  }

  return (
    <Section title="Pending Signals" badge={`${signals.length} pending`}>
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {signals.map((signal) => (
          <div
            key={signal.id}
            className={`p-4 rounded-lg border-2 ${
              signal.action === 'BUY'
                ? 'border-green-300 bg-green-50'
                : 'border-red-300 bg-red-50'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-lg">{signal.symbol.replace('_', '/')}</span>
                  <span
                    className={`px-2 py-0.5 rounded text-sm font-bold ${
                      signal.action === 'BUY'
                        ? 'bg-green-200 text-green-800'
                        : 'bg-red-200 text-red-800'
                    }`}
                  >
                    {signal.action}
                  </span>
                </div>
                <div className="flex gap-4 mt-1 text-sm text-gray-600">
                  <span>Confidence: {(signal.confidence * 100).toFixed(1)}%</span>
                  <span>Size: ${signal.recommended_size_usd?.toFixed(2) || 'N/A'}</span>
                  <span>Source: {signal.signal_source}</span>
                </div>
                {signal.expires_at && (
                  <div className="text-xs text-gray-500 mt-1">
                    Expires: {new Date(signal.expires_at).toLocaleTimeString()}
                  </div>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => onApprove(signal.id)}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
                >
                  Approve
                </button>
                <button
                  onClick={() => onReject(signal.id)}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 font-medium"
                >
                  Reject
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}

// Trading Settings Panel
function TradingSettingsPanel({ settings, onUpdate, loading }) {
  const [localSettings, setLocalSettings] = useState(settings || {})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (settings) setLocalSettings(settings)
  }, [settings])

  const handleSave = async () => {
    setSaving(true)
    try {
      await onUpdate(localSettings)
    } finally {
      setSaving(false)
    }
  }

  if (loading || !settings) return <LoadingCard message="Loading settings..." />

  return (
    <Section title="Trading Settings" badge={settings.paper_trading ? 'Paper' : 'Live'}>
      <div className="space-y-4">
        {/* Trading Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Trading Mode</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {TRADING_MODES.map((mode) => (
              <button
                key={mode.value}
                onClick={() => setLocalSettings({ ...localSettings, trading_mode: mode.value })}
                className={`p-3 rounded-lg border-2 text-left transition-all ${
                  localSettings.trading_mode === mode.value
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="font-medium">{mode.label}</div>
                <div className="text-xs text-gray-500">{mode.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Risk Profile */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Risk Profile</label>
          <div className="grid grid-cols-3 gap-2">
            {RISK_PROFILES.map((profile) => (
              <button
                key={profile.value}
                onClick={() => setLocalSettings({ ...localSettings, risk_profile: profile.value })}
                className={`p-3 rounded-lg border-2 text-left transition-all ${
                  localSettings.risk_profile === profile.value
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="font-medium">{profile.label}</div>
                <div className="text-xs text-gray-500">{profile.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Numeric Settings */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Auto-Execute Threshold
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={localSettings.min_confidence_auto || 0.75}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, min_confidence_auto: parseFloat(e.target.value) })
              }
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">Min confidence for auto-trade</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Position ($)
            </label>
            <input
              type="number"
              min="0"
              step="100"
              value={localSettings.max_position_size_usd || 1000}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, max_position_size_usd: parseFloat(e.target.value) })
              }
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">Max per trade</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Daily Loss Limit ($)
            </label>
            <input
              type="number"
              min="0"
              step="50"
              value={localSettings.max_daily_loss_usd || 500}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, max_daily_loss_usd: parseFloat(e.target.value) })
              }
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">Stop trading if reached</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Trades/Day
            </label>
            <input
              type="number"
              min="1"
              step="1"
              value={localSettings.max_trades_per_day || 20}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, max_trades_per_day: parseInt(e.target.value) })
              }
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Cooldown (minutes)
            </label>
            <input
              type="number"
              min="0"
              step="1"
              value={localSettings.cooldown_minutes || 5}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, cooldown_minutes: parseInt(e.target.value) })
              }
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">Between same symbol</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Signal Expiry (min)
            </label>
            <input
              type="number"
              min="1"
              step="1"
              value={localSettings.signal_expiry_minutes || 15}
              onChange={(e) =>
                setLocalSettings({ ...localSettings, signal_expiry_minutes: parseInt(e.target.value) })
              }
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Paper Trading Toggle */}
        <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
          <input
            type="checkbox"
            id="paper_trading"
            checked={localSettings.paper_trading !== false}
            onChange={(e) => setLocalSettings({ ...localSettings, paper_trading: e.target.checked })}
            className="w-5 h-5"
          />
          <label htmlFor="paper_trading" className="flex-1">
            <span className="font-medium">Paper Trading Mode</span>
            <p className="text-sm text-gray-600">
              {localSettings.paper_trading !== false
                ? 'Safe mode - no real money at risk'
                : 'LIVE MODE - Real trades will be executed!'}
            </p>
          </label>
        </div>

        {/* Save Button */}
        <button
          onClick={handleSave}
          disabled={saving}
          className={`w-full py-3 rounded-lg font-bold text-white transition-all ${
            saving ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </Section>
  )
}

// Signal History
function SignalHistory({ signals, loading }) {
  if (loading) return <LoadingCard message="Loading history..." />

  if (!signals || signals.length === 0) {
    return (
      <Section title="Signal History">
        <div className="text-center text-gray-500 py-4">No signal history yet</div>
      </Section>
    )
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'executed':
        return 'bg-green-100 text-green-800'
      case 'auto_approved':
        return 'bg-blue-100 text-blue-800'
      case 'rejected':
        return 'bg-red-100 text-red-800'
      case 'expired':
        return 'bg-gray-100 text-gray-800'
      case 'failed':
        return 'bg-orange-100 text-orange-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <Section title="Signal History" badge={`${signals.length} signals`}>
      <div className="max-h-64 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className="px-3 py-2 text-left">Symbol</th>
              <th className="px-3 py-2 text-left">Action</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-right">Confidence</th>
              <th className="px-3 py-2 text-right">Size</th>
              <th className="px-3 py-2 text-left">Time</th>
            </tr>
          </thead>
          <tbody>
            {signals.slice(0, 50).map((signal) => (
              <tr key={signal.id} className="border-b hover:bg-gray-50">
                <td className="px-3 py-2 font-medium">{signal.symbol.replace('_', '/')}</td>
                <td className="px-3 py-2">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium ${
                      signal.action === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {signal.action}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(signal.status)}`}>
                    {signal.status.replace('_', ' ')}
                  </span>
                </td>
                <td className="px-3 py-2 text-right">{(signal.confidence * 100).toFixed(0)}%</td>
                <td className="px-3 py-2 text-right">${signal.recommended_size_usd?.toFixed(0) || '-'}</td>
                <td className="px-3 py-2 text-gray-500">
                  {signal.actioned_at ? new Date(signal.actioned_at).toLocaleTimeString() : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  )
}

// Main Trading Dashboard Component
export default function TradingDashboard() {
  const [status, setStatus] = useState(null)
  const [settings, setSettings] = useState(null)
  const [dailyStats, setDailyStats] = useState(null)
  const [pendingSignals, setPendingSignals] = useState([])
  const [signalHistory, setSignalHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const [statusData, settingsData, statsData, pendingData, historyData] = await Promise.all([
        autonomousTradingAPI.getTradingStatus().catch(() => null),
        autonomousTradingAPI.getSettings().catch(() => null),
        autonomousTradingAPI.getDailyStats().catch(() => null),
        autonomousTradingAPI.getPendingSignals().catch(() => []),
        autonomousTradingAPI.getSignalHistory(50).catch(() => []),
      ])

      setStatus(statusData)
      setSettings(settingsData)
      setDailyStats(statsData)
      setPendingSignals(pendingData)
      setSignalHistory(historyData)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [fetchData])

  const handleActivateKillSwitch = async (reason) => {
    await autonomousTradingAPI.activateKillSwitch(reason)
    fetchData()
  }

  const handleDeactivateKillSwitch = async () => {
    await autonomousTradingAPI.deactivateKillSwitch()
    fetchData()
  }

  const handleUpdateSettings = async (newSettings) => {
    await autonomousTradingAPI.updateSettings(newSettings)
    fetchData()
  }

  const handleApproveSignal = async (signalId) => {
    try {
      await autonomousTradingAPI.approveSignal(signalId)
      fetchData()
    } catch (err) {
      alert(`Failed to approve: ${err.message}`)
    }
  }

  const handleRejectSignal = async (signalId) => {
    try {
      await autonomousTradingAPI.rejectSignal(signalId)
      fetchData()
    } catch (err) {
      alert(`Failed to reject: ${err.message}`)
    }
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-xl border p-6">
          <h2 className="text-xl font-bold mb-2">Autonomous Trading</h2>
          <p className="text-gray-600 text-sm">Loading trading system...</p>
        </div>
        <LoadingCard message="Initializing autonomous trading..." />
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-xl border p-6">
          <h2 className="text-xl font-bold mb-2">Autonomous Trading</h2>
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
            <p className="font-medium">Error loading trading system</p>
            <p className="text-sm mt-1">{error}</p>
            <button
              onClick={fetchData}
              className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl border p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold mb-1">Autonomous Trading System</h2>
            <p className="text-gray-600 text-sm">
              AI-powered trading with full user control and risk management
            </p>
          </div>
          <div className="flex items-center gap-2">
            {settings?.paper_trading !== false && (
              <span className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium">
                Paper Trading
              </span>
            )}
            {status?.kill_switch_active && (
              <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium">
                Kill Switch Active
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Kill Switch */}
      <KillSwitch
        active={settings?.kill_switch_active || false}
        reason={settings?.kill_switch_reason || ''}
        onActivate={handleActivateKillSwitch}
        onDeactivate={handleDeactivateKillSwitch}
      />

      {/* Status Overview */}
      <TradingStatus status={status} dailyStats={dailyStats} />

      {/* Main Content Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          <PendingSignalsQueue
            signals={pendingSignals}
            onApprove={handleApproveSignal}
            onReject={handleRejectSignal}
            loading={false}
          />
          <SignalHistory signals={signalHistory} loading={false} />
        </div>

        {/* Right Column */}
        <div>
          <TradingSettingsPanel
            settings={settings}
            onUpdate={handleUpdateSettings}
            loading={false}
          />
        </div>
      </div>
    </div>
  )
}
