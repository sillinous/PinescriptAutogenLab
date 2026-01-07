import React, { useEffect, useState } from 'react'
import { tradingAPI } from '../../services/api'
import { Section } from '../common/Section'
import { Card } from '../common/Card'
import { LoadingCard } from '../common/Loading'

// localStorage keys for user preferences
const STORAGE_KEYS = {
  AB_SYMBOL_A: 'pinelab_ab_symbol_a',
  AB_SYMBOL_B: 'pinelab_ab_symbol_b',
  AUTOTUNE_SYMBOL: 'pinelab_autotune_symbol',
}

// Symbol selector dropdown component
function SymbolSelector({ value, onChange, symbols, label, disabled = false }) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-gray-600 whitespace-nowrap">{label}:</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="text-xs border border-gray-300 rounded px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
      >
        {symbols.map((sym) => (
          <option key={sym} value={sym}>
            {sym.replace('_', '/')}
          </option>
        ))}
      </select>
    </div>
  )
}

export function ABTestingPanel({ availableSymbols = [] }) {
  const [ab, setAb] = useState(null)
  const [loading, setLoading] = useState(true)
  const [symbolA, setSymbolA] = useState(() =>
    localStorage.getItem(STORAGE_KEYS.AB_SYMBOL_A) || 'BTC_USDT'
  )
  const [symbolB, setSymbolB] = useState(() =>
    localStorage.getItem(STORAGE_KEYS.AB_SYMBOL_B) || 'ETH_USDT'
  )

  // Save preferences to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.AB_SYMBOL_A, symbolA)
  }, [symbolA])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.AB_SYMBOL_B, symbolB)
  }, [symbolB])

  // Fetch A/B status with selected symbols
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const data = await tradingAPI.getABStatus(symbolA, symbolB)
        setAb(data)
      } catch (err) {
        console.error('AB fetch error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 12000)
    return () => clearInterval(interval)
  }, [symbolA, symbolB])

  // Use available symbols or defaults
  const symbols = availableSymbols.length > 0
    ? availableSymbols
    : ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'DOGE_USDT', 'ADA_USDT']

  if (loading && !ab) return <LoadingCard message="Loading A/B tests..." />

  const symbolADisplay = symbolA.replace('_', '/')
  const symbolBDisplay = symbolB.replace('_', '/')

  return (
    <Section
      title="A/B Live Shadow Deployments"
      badge="Active"
      action={
        <div className="flex items-center gap-3">
          <SymbolSelector
            value={symbolA}
            onChange={setSymbolA}
            symbols={symbols.filter(s => s !== symbolB)}
            label="Symbol A"
          />
          <span className="text-gray-400">vs</span>
          <SymbolSelector
            value={symbolB}
            onChange={setSymbolB}
            symbols={symbols.filter(s => s !== symbolA)}
            label="Symbol B"
          />
        </div>
      }
    >
      {ab ? (
        <div className="space-y-3">
          <div className="text-sm">
            <span className="text-gray-600">Test:</span>{' '}
            <span className="font-semibold">{ab.test_name}</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className={`p-4 rounded-lg border-2 ${ab.winner === 'A' ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-gray-50'}`}>
              <div className="text-xs text-gray-600 mb-1">Variant A ({symbolADisplay})</div>
              <div className="text-2xl font-bold">{ab.variant_a_winrate}%</div>
              <div className="text-xs text-gray-500">Win Rate</div>
              {ab.returns && (
                <div className={`text-xs mt-1 ${ab.returns.symbol_a >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {ab.returns.symbol_a >= 0 ? '+' : ''}{ab.returns.symbol_a}% return
                </div>
              )}
            </div>
            <div className={`p-4 rounded-lg border-2 ${ab.winner === 'B' ? 'border-green-500 bg-green-50' : 'border-gray-200 bg-gray-50'}`}>
              <div className="text-xs text-gray-600 mb-1">Variant B ({symbolBDisplay})</div>
              <div className="text-2xl font-bold">{ab.variant_b_winrate}%</div>
              <div className="text-xs text-gray-500">Win Rate</div>
              {ab.returns && (
                <div className={`text-xs mt-1 ${ab.returns.symbol_b >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {ab.returns.symbol_b >= 0 ? '+' : ''}{ab.returns.symbol_b}% return
                </div>
              )}
            </div>
          </div>
          <div className={`rounded-lg p-3 text-center ${
            ab.winner === 'TIE'
              ? 'bg-gray-100 border border-gray-300'
              : 'bg-blue-50 border border-blue-200'
          }`}>
            <div className="text-xs text-gray-600 mb-1">Current Winner</div>
            <div className={`text-2xl font-bold ${
              ab.winner === 'TIE' ? 'text-gray-600' : 'text-blue-700'
            }`}>
              {ab.winner === 'TIE' ? 'TIE' : ab.winner === 'A' ? symbolADisplay : symbolBDisplay}
            </div>
          </div>
          {ab.error && (
            <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded">
              Note: {ab.error}
            </div>
          )}
        </div>
      ) : (
        <div className="text-center text-gray-500 py-8">No A/B tests running</div>
      )}
    </Section>
  )
}

export function AutoOptimizationPanel({ availableSymbols = [] }) {
  const [tune, setTune] = useState(null)
  const [loading, setLoading] = useState(true)
  const [symbol, setSymbol] = useState(() =>
    localStorage.getItem(STORAGE_KEYS.AUTOTUNE_SYMBOL) || 'BTC_USDT'
  )

  // Save preference to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.AUTOTUNE_SYMBOL, symbol)
  }, [symbol])

  // Fetch autotune status with selected symbol
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const data = await tradingAPI.getAutotuneStatus(symbol)
        setTune(data)
      } catch (err) {
        console.error('Autotune fetch error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 12000)
    return () => clearInterval(interval)
  }, [symbol])

  // Use available symbols or defaults
  const symbols = availableSymbols.length > 0
    ? availableSymbols
    : ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'XRP_USDT', 'DOGE_USDT', 'ADA_USDT']

  if (loading && !tune) return <LoadingCard message="Loading optimization..." />

  const symbolDisplay = symbol.replace('_', '/')

  return (
    <Section
      title="Auto-Optimization Engine"
      badge="Optuna"
      action={
        <SymbolSelector
          value={symbol}
          onChange={setSymbol}
          symbols={symbols}
          label="Analyze"
        />
      }
    >
      {tune ? (
        <div className="space-y-3">
          <div className="text-sm">
            <span className="text-gray-600">Optimizing:</span>{' '}
            <span className="font-semibold">{symbolDisplay}</span>
          </div>
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
            <div className="text-xs text-gray-600 mb-2">Best Parameters for {symbolDisplay}</div>
            <pre className="text-xs font-mono bg-white p-2 rounded border overflow-auto max-h-32">
              {JSON.stringify(tune.best_parameters, null, 2)}
            </pre>
          </div>
          {tune.error && (
            <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded">
              Note: {tune.error}
            </div>
          )}
        </div>
      ) : (
        <div className="text-center text-gray-500 py-8">No optimization running</div>
      )}
    </Section>
  )
}
