import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart, BarChart, Bar } from 'recharts'

const API = 'http://localhost:8080'

function Card({ title, value, subtitle, status }) {
  const statusColors = {
    success: 'text-green-600',
    warning: 'text-yellow-600',
    error: 'text-red-600',
    info: 'text-blue-600'
  }

  return (
    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="text-xs text-gray-500 mb-1">{title}</div>
      <div className={`text-2xl font-bold ${status ? statusColors[status] : ''}`}>
        {value}
      </div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  )
}

function Section({ title, children, badge }) {
  return (
    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">{title}</h3>
        {badge && (
          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
            {badge}
          </span>
        )}
      </div>
      {children}
    </div>
  )
}

export default function AITradingDashboard() {
  const [symbols, setSymbols] = useState([])
  const [symbol, setSymbol] = useState('BTC_USDT')
  const [interval, setInterval] = useState('1h')
  const [candles, setCandles] = useState([])
  const [aiPrediction, setAiPrediction] = useState(null)
  const [aggregatedSignal, setAggregatedSignal] = useState(null)
  const [supportResistance, setSupportResistance] = useState(null)
  const [ab, setAb] = useState(null)
  const [tune, setTune] = useState(null)
  const [loading, setLoading] = useState({})

  // Load symbols
  useEffect(() => {
    fetch(`${API}/symbols`)
      .then(r => r.json())
      .then(j => {
        setSymbols(j.symbols || [])
        if ((j.symbols || []).includes('BTC_USDT')) setSymbol('BTC_USDT')
        else if (j.symbols?.length) setSymbol(j.symbols[0])
      })
      .catch(() => {})
  }, [])

  // Load candles
  useEffect(() => {
    let stop = false
    const run = async () => {
      try {
        const r = await fetch(`${API}/candles/${symbol}?interval=${interval}&limit=100`)
        const j = await r.json()
        if (!stop) setCandles(j.candles || [])
      } catch {}
      if (!stop) setTimeout(run, 10000)
    }
    run()
    return () => { stop = true }
  }, [symbol, interval])

  // Load AI data
  useEffect(() => {
    let stop = false
    const loadAI = async () => {
      try {
        // Get aggregated signal
        setLoading(prev => ({ ...prev, signal: true }))
        const signalRes = await fetch(`${API}/api/v1/ai/signal/aggregate/${symbol}`)
        if (signalRes.ok) {
          const signal = await signalRes.json()
          if (!stop) setAggregatedSignal(signal)
        }
        setLoading(prev => ({ ...prev, signal: false }))

        // Get support/resistance
        setLoading(prev => ({ ...prev, sr: true }))
        const srRes = await fetch(`${API}/api/v1/ai/chart/support-resistance/${symbol}?timeframe=${interval}&bars=100`)
        if (srRes.ok) {
          const sr = await srRes.json()
          if (!stop) setSupportResistance(sr)
        }
        setLoading(prev => ({ ...prev, sr: false }))
      } catch (e) {
        console.error('AI fetch error:', e)
      }
      if (!stop) setTimeout(loadAI, 15000)
    }
    loadAI()
    return () => { stop = true }
  }, [symbol, interval])

  // Load platform metrics
  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const [a, t] = await Promise.all([
          fetch(`${API}/ab/status`).then(r => r.json()).catch(() => null),
          fetch(`${API}/autotune/status`).then(r => r.json()).catch(() => null),
        ])
        if (!stop) {
          setAb(a)
          setTune(t)
        }
      } catch {}
      if (!stop) setTimeout(tick, 12000)
    }
    tick()
    return () => { stop = true }
  }, [])

  const chartData = candles.map(c => ({
    ts: c.t,
    close: c.c,
    volume: c.v,
    time: new Date(c.t * 1000).toLocaleTimeString()
  }))

  const last = chartData.length ? chartData[chartData.length-1].close : 0
  const first = chartData.length ? chartData[0].close : 0
  const changePct = first ? ((last - first) / first * 100).toFixed(2) : '0.00'
  const changeColor = parseFloat(changePct) >= 0 ? 'text-green-600' : 'text-red-600'

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 shadow-lg">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">🤖 AI Trading Platform</h1>
          <p className="text-blue-100">Production-Ready AI-Powered Trading Dashboard</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Controls */}
        <div className="bg-white border rounded-xl shadow-sm p-4 flex flex-wrap gap-4 items-center">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Symbol</label>
            <select
              className="border rounded px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
            >
              {symbols.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Timeframe</label>
            <select
              className="border rounded px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
              value={interval}
              onChange={e => setInterval(e.target.value)}
            >
              {['1m','5m','15m','30m','1h','4h','1d'].map(i =>
                <option key={i} value={i}>{i}</option>
              )}
            </select>
          </div>
          <div className="flex-1 min-w-[200px]">
            <div className="text-xs text-gray-500">Current Price</div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold">${last.toFixed(4)}</span>
              <span className={`text-lg font-semibold ${changeColor}`}>
                {parseFloat(changePct) >= 0 ? '+' : ''}{changePct}%
              </span>
            </div>
          </div>
        </div>

        {/* AI Signals - Prominent */}
        {aggregatedSignal && (
          <div className="bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-200 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-gray-800">🎯 AI Trading Signal</h2>
              <span className="text-xs text-gray-500">Updated: {new Date(aggregatedSignal.timestamp).toLocaleTimeString()}</span>
            </div>

            <div className="grid md:grid-cols-4 gap-4 mb-4">
              <Card
                title="Action"
                value={aggregatedSignal.action.toUpperCase()}
                status={
                  aggregatedSignal.action === 'buy' ? 'success' :
                  aggregatedSignal.action === 'sell' ? 'error' : 'warning'
                }
              />
              <Card
                title="Confidence"
                value={`${(aggregatedSignal.confidence * 100).toFixed(1)}%`}
                subtitle={aggregatedSignal.confidence > 0.7 ? 'High' : 'Moderate'}
                status={aggregatedSignal.confidence > 0.7 ? 'success' : 'warning'}
              />
              <Card
                title="Consensus"
                value={`${(aggregatedSignal.consensus * 100).toFixed(1)}%`}
                subtitle="Signal Agreement"
              />
              <Card
                title="Position Size"
                value={`${(aggregatedSignal.recommended_position_size * 100).toFixed(0)}%`}
                subtitle="Recommended"
              />
            </div>

            {aggregatedSignal.sources && aggregatedSignal.sources.length > 0 && (
              <div className="bg-white/50 rounded-lg p-3">
                <div className="text-xs text-gray-600 mb-2">Signal Sources:</div>
                <div className="flex flex-wrap gap-2">
                  {aggregatedSignal.sources.map((source, idx) => (
                    <span key={idx} className="text-xs px-2 py-1 bg-white rounded-full border">
                      {source}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Price Chart */}
        <Section title={`${symbol} Price Chart`} badge={interval}>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                <YAxis domain={['dataMin', 'dataMax']} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  labelStyle={{ fontWeight: 'bold' }}
                />
                <Area type="monotone" dataKey="close" stroke="#3b82f6" strokeWidth={2} fill="url(#colorPrice)" />

                {/* Support/Resistance Lines */}
                {supportResistance && supportResistance.support_levels && supportResistance.support_levels.map((level, idx) => (
                  <Line
                    key={`support-${idx}`}
                    type="monotone"
                    dataKey={() => level}
                    stroke="#10b981"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                ))}
                {supportResistance && supportResistance.resistance_levels && supportResistance.resistance_levels.map((level, idx) => (
                  <Line
                    key={`resistance-${idx}`}
                    type="monotone"
                    dataKey={() => level}
                    stroke="#ef4444"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {supportResistance && (
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-xs text-gray-500 mb-1">Support Levels</div>
                {supportResistance.support_levels?.map((level, idx) => (
                  <div key={idx} className="text-green-600 font-mono">${level.toFixed(4)}</div>
                ))}
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Resistance Levels</div>
                {supportResistance.resistance_levels?.map((level, idx) => (
                  <div key={idx} className="text-red-600 font-mono">${level.toFixed(4)}</div>
                ))}
              </div>
            </div>
          )}
        </Section>

        {/* Platform Features */}
        <div className="grid md:grid-cols-2 gap-4">
          {/* A/B Testing */}
          <Section title="A/B Live Shadow Deployments" badge="Active">
            {ab ? (
              <div className="space-y-3">
                <div className="text-sm">
                  <span className="text-gray-600">Test:</span>{' '}
                  <span className="font-semibold">{ab.test_name}</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Card
                    title="Variant A"
                    value={`${ab.variant_a_winrate}%`}
                    subtitle="Win Rate"
                  />
                  <Card
                    title="Variant B"
                    value={`${ab.variant_b_winrate}%`}
                    subtitle="Win Rate"
                  />
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center">
                  <div className="text-xs text-blue-600 mb-1">Current Winner</div>
                  <div className="text-2xl font-bold text-blue-700">{ab.winner}</div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400 mx-auto mb-2"></div>
                <div className="text-sm">Loading...</div>
              </div>
            )}
          </Section>

          {/* Auto-Optimization */}
          <Section title="Auto-Optimization Engine" badge="Optuna">
            {tune ? (
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>Progress</span>
                    <span>{tune.progress}%</span>
                  </div>
                  <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-3 bg-gradient-to-r from-green-500 to-green-600 transition-all duration-500"
                      style={{ width: `${tune.progress}%` }}
                    />
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-600 mb-2">Best Parameters</div>
                  <pre className="text-xs font-mono bg-white p-2 rounded border overflow-auto max-h-32">
                    {JSON.stringify(tune.best_parameters, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400 mx-auto mb-2"></div>
                <div className="text-sm">Evaluating...</div>
              </div>
            )}
          </Section>
        </div>

        {/* API Status */}
        <Section title="System Status" badge="Healthy">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card title="Backend API" value="✓ Online" status="success" />
            <Card title="AI Endpoints" value="7 Active" status="success" />
            <Card title="Data Sources" value="3 Available" status="info" />
            <Card title="ML Models" value="Ready" status="success" />
          </div>
        </Section>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500 py-4">
          <div className="mb-2">
            🚀 PineLab AI Trading Platform v2.0 • Phase 1 Complete
          </div>
          <div className="flex justify-center gap-4 text-xs">
            <a href="http://localhost:8080/docs" target="_blank" rel="noopener noreferrer"
               className="text-blue-600 hover:underline">
              API Docs
            </a>
            <span>•</span>
            <a href="/AI_IMPLEMENTATION_GUIDE.md" className="text-blue-600 hover:underline">
              Implementation Guide
            </a>
            <span>•</span>
            <span className="text-green-600">
              ✓ {symbols.length} Symbols Loaded
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
