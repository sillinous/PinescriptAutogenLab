import React, { useEffect, useState } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { tradingAPI, aiAPI } from '../../services/api'
import { Section } from '../common/Section'
import { LoadingOverlay } from '../common/Loading'

export function AdvancedPriceChart({ symbol, interval = '1h', showSR = true }) {
  const [candles, setCandles] = useState([])
  const [srLevels, setSRLevels] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true

    const fetchData = async () => {
      try {
        setLoading(true)

        // Fetch candles
        const candleData = await tradingAPI.getCandles(symbol, interval, 100)
        if (mounted) setCandles(candleData.candles || [])

        // Fetch support/resistance if enabled
        if (showSR) {
          try {
            const srData = await aiAPI.getSupportResistance(symbol, interval, 100)
            if (mounted) setSRLevels(srData)
          } catch (err) {
            console.warn('Could not fetch S/R levels:', err.message)
          }
        }
      } catch (err) {
        console.error('Chart error:', err)
      } finally {
        if (mounted) setLoading(false)
      }
    }

    fetchData()
    const interval_id = setInterval(fetchData, 30000)

    return () => {
      mounted = false
      clearInterval(interval_id)
    }
  }, [symbol, interval, showSR])

  const chartData = candles.map((c) => ({
    timestamp: c.t,
    time: new Date(c.t * 1000).toLocaleTimeString(),
    price: c.c,
    volume: c.v,
    high: c.h,
    low: c.l,
  }))

  const lastPrice = chartData.length ? chartData[chartData.length - 1].price : 0
  const firstPrice = chartData.length ? chartData[0].price : 0
  const changePct = firstPrice ? ((lastPrice - firstPrice) / firstPrice) * 100 : 0
  const changeColor = changePct >= 0 ? 'text-green-600' : 'text-red-600'

  return (
    <Section
      key={symbol}
      title={`${symbol} Price Chart`}
      badge={interval}
      action={
        <div className="flex items-baseline gap-2 text-sm">
          <span className="font-mono font-bold">${lastPrice.toFixed(4)}</span>
          <span className={`font-semibold ${changeColor}`}>
            {changePct >= 0 ? '+' : ''}
            {changePct.toFixed(2)}%
          </span>
        </div>
      }
    >
      <div className="relative">
        {loading && <LoadingOverlay message="Loading chart..." />}

        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis
                domain={['dataMin - 10', 'dataMax + 10']}
                tick={{ fontSize: 11 }}
                tickFormatter={(value) => `$${value.toFixed(2)}`}
              />
              <Tooltip
                contentStyle={{
                  background: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
                formatter={(value) => [`$${value.toFixed(4)}`, 'Price']}
              />

              <Area
                type="monotone"
                dataKey="price"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#priceGradient)"
              />

              {/* Support Levels */}
              {showSR &&
                srLevels?.support_levels?.map((level, idx) => (
                  <ReferenceLine
                    key={`support-${idx}`}
                    y={level}
                    stroke="#10b981"
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    label={{
                      value: `S: $${level.toFixed(2)}`,
                      fill: '#10b981',
                      fontSize: 11,
                    }}
                  />
                ))}

              {/* Resistance Levels */}
              {showSR &&
                srLevels?.resistance_levels?.map((level, idx) => (
                  <ReferenceLine
                    key={`resistance-${idx}`}
                    y={level}
                    stroke="#ef4444"
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    label={{
                      value: `R: $${level.toFixed(2)}`,
                      fill: '#ef4444',
                      fontSize: 11,
                    }}
                  />
                ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* S/R Legend */}
        {showSR && srLevels && (
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-0.5 bg-green-500 border-green-500 border-dashed"></div>
                <span className="text-xs font-medium text-gray-700">Support Levels</span>
              </div>
              {srLevels.support_levels?.map((level, idx) => (
                <div key={idx} className="text-green-600 font-mono text-xs">
                  ${level.toFixed(4)}
                </div>
              ))}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-0.5 bg-red-500 border-red-500 border-dashed"></div>
                <span className="text-xs font-medium text-gray-700">Resistance Levels</span>
              </div>
              {srLevels.resistance_levels?.map((level, idx) => (
                <div key={idx} className="text-red-600 font-mono text-xs">
                  ${level.toFixed(4)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  )
}
