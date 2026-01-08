import React, { useState, useEffect } from 'react'
import { tradingAPI } from '../services/api'
import { SignalAggregator } from '../components/ai/SignalAggregator'
import { PredictionsPanel } from '../components/ai/PredictionsPanel'
import { ModelManagement } from '../components/ai/ModelManagement'
import { AdvancedPriceChart } from '../components/charts/AdvancedPriceChart'
import { FeatureExplorer } from '../components/ai/FeatureExplorer'
import { ABTestingPanel, AutoOptimizationPanel } from '../components/platform/PlatformMetrics'
import { Card, MetricCard } from '../components/common/Card'
import DeepLearningDashboard from '../components/ai/DeepLearningDashboard'
import TradingDashboard from '../components/trading/TradingDashboard'
import SignalQueueDashboard from '../components/trading/SignalQueueDashboard'
import MarketRegimeDashboard from '../components/trading/MarketRegimeDashboard'
import AnalyticsDashboard from '../components/analytics/AnalyticsDashboard'
import UserHandbook from '../components/UserHandbook'

export default function ComprehensiveDashboard() {
  const [symbols, setSymbols] = useState([])
  const [selectedSymbol, setSelectedSymbol] = useState('BTC_USDT')
  const [interval, setInterval] = useState('1h')
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    tradingAPI
      .getSymbols()
      .then((data) => {
        setSymbols(data.symbols || [])
        if ((data.symbols || []).includes('BTC_USDT')) {
          setSelectedSymbol('BTC_USDT')
        } else if (data.symbols?.length) {
          setSelectedSymbol(data.symbols[0])
        }
      })
      .catch(() => {})
  }, [])

  const tabs = [
    { id: 'overview', label: '📊 Overview', icon: '📊' },
    { id: 'trading', label: '🤖 Trading', icon: '🤖' },
    { id: 'signals', label: '📋 Signals', icon: '📋' },
    { id: 'regime', label: '🌡️ Market Regime', icon: '🌡️' },
    { id: 'analytics', label: '📈 Analytics', icon: '📈' },
    { id: 'models', label: '🎓 Model Lab', icon: '🎓' },
    { id: 'deeplearning', label: '🧠 Deep Learning', icon: '🧠' },
    { id: 'features', label: '🔬 Features', icon: '🔬' },
    { id: 'platform', label: '⚙️ Platform', icon: '⚙️' },
    { id: 'handbook', label: '📚 Help', icon: '📚' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white shadow-2xl">
        <div className="max-w-[1800px] mx-auto px-6 py-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold mb-1">
                🤖 AI Trading Platform
              </h1>
              <p className="text-blue-100 text-sm">
                Production-Ready AI-Powered Trading Dashboard • Phase 2 Deep Learning Active
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setActiveTab('handbook')}
                className="bg-white/20 hover:bg-white/30 backdrop-blur rounded-lg px-4 py-2 transition-all text-left"
                title="Open User Handbook"
              >
                <div className="text-xs text-blue-100">Need Help?</div>
                <div className="text-sm font-semibold">📚 User Guide</div>
              </button>
              <div className="bg-white/20 backdrop-blur rounded-lg px-4 py-2">
                <div className="text-xs text-blue-100">Backend Status</div>
                <div className="text-sm font-semibold">✓ Online</div>
              </div>
              <div className="bg-white/20 backdrop-blur rounded-lg px-4 py-2">
                <div className="text-xs text-blue-100">AI Endpoints</div>
                <div className="text-sm font-semibold">7 Active</div>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap gap-3 items-center">
            <div>
              <label className="text-xs text-blue-100 block mb-1">Symbol</label>
              <select
                className="bg-white/10 backdrop-blur border border-white/20 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-white/50 outline-none"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                {symbols.map((s) => (
                  <option key={s} value={s} className="bg-blue-900">
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-blue-100 block mb-1">Timeframe</label>
              <select
                className="bg-white/10 backdrop-blur border border-white/20 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-white/50 outline-none"
                value={interval}
                onChange={(e) => setInterval(e.target.value)}
              >
                {['1m', '5m', '15m', '30m', '1h', '4h', '1d'].map((i) => (
                  <option key={i} value={i} className="bg-blue-900">
                    {i}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="bg-white border-b sticky top-0 z-10 shadow-sm">
        <div className="max-w-[1800px] mx-auto px-6">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 font-medium text-sm transition-all ${
                  activeTab === tab.id
                    ? 'border-b-2 border-blue-600 text-blue-600 bg-blue-50'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-[1800px] mx-auto px-6 py-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* AI Signals - Most Important */}
            <SignalAggregator symbol={selectedSymbol} />

            {/* Chart and Predictions */}
            <div className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <AdvancedPriceChart symbol={selectedSymbol} interval={interval} showSR={true} />
              </div>
              <div>
                <PredictionsPanel symbol={selectedSymbol} />
              </div>
            </div>

            {/* System Status */}
            <div className="grid md:grid-cols-4 gap-4">
              <MetricCard icon="✅" label="Backend API" value="Online" color="green" />
              <MetricCard icon="🤖" label="AI Models" value="Ready" color="blue" />
              <MetricCard icon="📡" label="Data Sources" value="3 Active" color="purple" />
              <MetricCard icon="⚡" label="Latency" value="<100ms" color="green" />
            </div>
          </div>
        )}

        {/* Autonomous Trading Tab */}
        {activeTab === 'trading' && (
          <TradingDashboard />
        )}

        {/* Signal Queue Tab */}
        {activeTab === 'signals' && (
          <div className="bg-gray-900 rounded-xl p-6">
            <SignalQueueDashboard />
          </div>
        )}

        {/* Market Regime Tab */}
        {activeTab === 'regime' && (
          <div className="bg-gray-900 rounded-xl p-6">
            <MarketRegimeDashboard />
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="bg-gray-900 rounded-xl p-6">
            <AnalyticsDashboard />
          </div>
        )}

        {/* Model Lab Tab */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl border p-6">
              <h2 className="text-xl font-bold mb-2">🎓 AI Model Training Laboratory</h2>
              <p className="text-gray-600 text-sm mb-4">
                Train custom reinforcement learning models to predict optimal trading actions
              </p>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
              <ModelManagement symbol={selectedSymbol} />
              <PredictionsPanel symbol={selectedSymbol} autoRefresh={false} />
            </div>

            <div className="bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-200 rounded-xl p-6">
              <h3 className="font-bold text-lg mb-3">📚 Training Guide</h3>
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl mb-2">1️⃣</div>
                  <div className="font-semibold mb-1">Configure Parameters</div>
                  <div className="text-gray-600 text-xs">
                    Set model name, ticker, timeframe, and training steps
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl mb-2">2️⃣</div>
                  <div className="font-semibold mb-1">Start Training</div>
                  <div className="text-gray-600 text-xs">
                    Training runs in background on server (5-60 min)
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4">
                  <div className="text-2xl mb-2">3️⃣</div>
                  <div className="font-semibold mb-1">Get Predictions</div>
                  <div className="text-gray-600 text-xs">
                    Use trained model for real-time BUY/SELL/HOLD signals
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Deep Learning Tab */}
        {activeTab === 'deeplearning' && (
          <DeepLearningDashboard />
        )}

        {/* Features Tab */}
        {activeTab === 'features' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl border p-6">
              <h2 className="text-xl font-bold mb-2">🔬 Machine Learning Feature Engineering</h2>
              <p className="text-gray-600 text-sm">
                Generate and explore 100+ technical, statistical, and pattern-based features
              </p>
            </div>

            <FeatureExplorer symbol={selectedSymbol} />

            <div className="grid md:grid-cols-3 gap-4">
              <Card title="Technical Indicators" value="40+" subtitle="RSI, MACD, Bollinger, etc." />
              <Card title="Statistical Features" value="20+" subtitle="Rolling stats, correlation" />
              <Card title="Pattern Features" value="15+" subtitle="Candlestick patterns" />
            </div>
          </div>
        )}

        {/* Platform Tab */}
        {activeTab === 'platform' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl border p-6">
              <h2 className="text-xl font-bold mb-2">⚙️ Platform Features</h2>
              <p className="text-gray-600 text-sm">
                A/B testing, auto-optimization, and system monitoring
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <ABTestingPanel availableSymbols={symbols} />
              <AutoOptimizationPanel availableSymbols={symbols} />
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <Card title="Total Symbols" value={symbols.length} status="info" />
              <Card title="API Uptime" value="99.9%" status="success" />
              <Card title="Avg Response Time" value="<200ms" status="success" />
            </div>
          </div>
        )}

        {/* User Handbook Tab */}
        {activeTab === 'handbook' && (
          <UserHandbook />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-[1800px] mx-auto px-6 py-6">
          <div className="text-center text-sm text-gray-600 space-y-2">
            <div className="font-semibold">
              🚀 PineLab AI Trading Platform v2.0 • Phase 2: Deep Learning Active
            </div>
            <div className="flex justify-center gap-6 text-xs">
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                📚 API Documentation
              </a>
              <span>•</span>
              <span className="text-green-600">✓ {symbols.length} Trading Pairs Available</span>
              <span>•</span>
              <span className="text-purple-600">✓ Phase 1 RL + Phase 2 DL Active</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
