
import React, { useEffect, useMemo, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart } from 'recharts'

const API = 'http://localhost:8080'

function fmtTs(ts) {
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString()
}

export default function Dashboard() {
  const [symbols, setSymbols] = useState([])
  const [symbol, setSymbol] = useState('BTC_USDT')
  const [interval, setInterval] = useState('1m')
  const [candles, setCandles] = useState([])
  const [summary, setSummary] = useState([])
  const [ab, setAb] = useState(null)
  const [tune, setTune] = useState(null)

  useEffect(() => {
    fetch(`${API}/symbols`).then(r => r.json()).then(j => {
      setSymbols(j.symbols || [])
      if ((j.symbols || []).includes('BTC_USDT')) setSymbol('BTC_USDT')
      else if (j.symbols?.length) setSymbol(j.symbols[0])
    })
  }, [])

  useEffect(() => {
    let stop = false
    const run = async () => {
      try {
        const r = await fetch(`${API}/candles/${symbol}?interval=${interval}&limit=200`)
        const j = await r.json()
        if (!stop) setCandles(j.candles || [])
      } catch {}
      if (!stop) setTimeout(run, 10000)
    }
    run()
    return () => { stop = true }
  }, [symbol, interval])

  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const [s, a, t] = await Promise.all([
          fetch(`${API}/summary`).then(r => r.json()),
          fetch(`${API}/ab/status`).then(r => r.json()),
          fetch(`${API}/autotune/status`).then(r => r.json()),
        ])
        if (!stop) {
          setSummary(s.tickers || [])
          setAb(a)
          setTune(t)
        }
      } catch {}
      if (!stop) setTimeout(tick, 12000)
    }
    tick()
    return () => { stop = true }
  }, [])

  const data = useMemo(() => candles.map(c => ({ ts: c.t, close: c.c, volume: c.v })), [candles])
  const last = data.length ? data[data.length-1].close : 0
  const first = data.length ? data[0].close : 0
  const changePct = first ? ((last - first) / first * 100).toFixed(2) : '0.00'

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-2 items-center">
        <select className="border rounded px-2 py-1" value={symbol} onChange={e => setSymbol(e.target.value)}>
          {symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select className="border rounded px-2 py-1" value={interval} onChange={e => setInterval(e.target.value)}>
          {['1m','5m','15m','30m','1h','4h','1d'].map(i => <option key={i} value={i}>{i}</option>)}
        </select>
        <div className="text-sm text-gray-600">Last: <span className="font-semibold">${last.toFixed(4)}</span> ({changePct}%)</div>
      </div>

      <div className="bg-white border rounded-xl shadow-sm p-4">
        <div className="text-sm font-semibold mb-2">Price — {symbol} ({interval})</div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <defs>
                <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.4}/>
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ts" tickFormatter={fmtTs} />
              <YAxis domain={['dataMin', 'dataMax']} />
              <Tooltip labelFormatter={(v) => new Date(v*1000).toLocaleString()} />
              <Area type="monotone" dataKey="close" stroke="#10b981" fill="url(#grad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-white border rounded-xl shadow-sm p-4">
          <div className="text-sm font-semibold mb-2">A/B Live Shadow Deployments</div>
          {ab ? (
            <div className="text-sm space-y-1">
              <div>Test: <span className="font-semibold">{ab.test_name}</span></div>
              <div>Variant A Winrate: <span className="font-semibold">{ab.variant_a_winrate}%</span></div>
              <div>Variant B Winrate: <span className="font-semibold">{ab.variant_b_winrate}%</span></div>
              <div>Winner: <span className="font-semibold">{ab.winner}</span></div>
            </div>
          ) : <div className="text-sm text-gray-500">Loading…</div>}
        </div>

        <div className="bg-white border rounded-xl shadow-sm p-4">
          <div className="text-sm font-semibold mb-2">Auto-Optimization</div>
          {tune ? (
            <div className="text-sm space-y-2">
              <div className="text-gray-600">Progress</div>
              <div className="w-full h-2 bg-gray-200 rounded">
                <div className="h-2 bg-green-600 rounded" style={{width: `${tune.progress}%`, transition: 'width .4s'}}/>
              </div>
              <div className="text-[11px] text-gray-500">{tune.progress}%</div>
              <div className="text-gray-600">Best Parameters</div>
              <pre className="text-[11px] bg-gray-50 p-2 rounded border">{JSON.stringify(tune.best_parameters, null, 2)}</pre>
            </div>
          ) : <div className="text-sm text-gray-500">Evaluating…</div>}
        </div>
      </div>

      <div className="bg-white border rounded-xl shadow-sm p-4">
        <div className="text-sm font-semibold mb-2">Market Summary (top)</div>
        <div className="grid md:grid-cols-4 gap-2">
          {(summary || []).slice(0, 12).map((t, idx) => (
            <div key={idx} className="border rounded p-2 text-sm">
              <div className="text-gray-500">{t.symbol}</div>
              <div className="font-semibold">${Number(t.last || 0).toFixed(6)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
