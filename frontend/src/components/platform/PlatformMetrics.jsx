import React, { useEffect, useState } from 'react'
import { tradingAPI } from '../../services/api'
import { Section } from '../common/Section'
import { Card } from '../common/Card'
import { LoadingCard } from '../common/Loading'

export function ABTestingPanel() {
  const [ab, setAb] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch = async () => {
      try {
        const data = await tradingAPI.getABStatus()
        setAb(data)
      } catch (err) {
        console.error('AB fetch error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetch()
    const interval = setInterval(fetch, 12000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <LoadingCard message="Loading A/B tests..." />

  return (
    <Section title="🔬 A/B Live Shadow Deployments" badge="Active">
      {ab ? (
        <div className="space-y-3">
          <div className="text-sm">
            <span className="text-gray-600">Test:</span>{' '}
            <span className="font-semibold">{ab.test_name}</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Card title="Variant A" value={`${ab.variant_a_winrate}%`} subtitle="Win Rate" />
            <Card title="Variant B" value={`${ab.variant_b_winrate}%`} subtitle="Win Rate" />
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center">
            <div className="text-xs text-blue-600 mb-1">Current Winner</div>
            <div className="text-2xl font-bold text-blue-700">{ab.winner}</div>
          </div>
        </div>
      ) : (
        <div className="text-center text-gray-500 py-8">No A/B tests running</div>
      )}
    </Section>
  )
}

export function AutoOptimizationPanel() {
  const [tune, setTune] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch = async () => {
      try {
        const data = await tradingAPI.getAutotuneStatus()
        setTune(data)
      } catch (err) {
        console.error('Autotune fetch error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetch()
    const interval = setInterval(fetch, 12000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <LoadingCard message="Loading optimization..." />

  return (
    <Section title="⚡ Auto-Optimization Engine" badge="Optuna">
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
        <div className="text-center text-gray-500 py-8">No optimization running</div>
      )}
    </Section>
  )
}
