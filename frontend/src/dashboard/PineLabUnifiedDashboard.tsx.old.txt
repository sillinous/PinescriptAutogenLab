
import React, { useState, useEffect } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, BarChart, Bar } from 'recharts';

const NavButton = ({ active, label, onClick }: any) => (
  <button onClick={onClick} className={`${active ? 'bg-black text-white' : 'bg-gray-200'} px-3 py-2 rounded-xl text-sm font-medium`}>{label}</button>
);

const MetricCard = ({ title, value, change, good }: any) => (
  <div className="flex flex-col p-4 bg-white rounded-xl border shadow-sm">
    <div className="text-sm text-gray-600">{title}</div>
    <div className="text-xl font-semibold">{value}</div>
    {change && (<div className={`text-xs ${good ? 'text-green-600' : 'text-red-500'}`}>{change}</div>)}
  </div>
);

function ABLiver({ apiBase }: { apiBase: string }) {
  const [abOn, setAbOn] = useState(false);
  const [cfg, setCfg] = useState({ control: 'live', shadow: 'paper', windowMins: 60, promoteIf: { sharpeDelta: 0.2, ddCap: 10 } });
  const [status, setStatus] = useState<any>({ state: 'idle', windowEnds: null, control: {}, candidate: {} });
  useEffect(()=>{
    let t:any; const poll = async ()=>{ try{
      const s = await fetch(`${apiBase}/ab/status`).then(r=>r.json());
      setStatus(s); setAbOn(!!s.enabled);
    }catch{} t=setTimeout(poll,15000)}; poll(); return ()=>clearTimeout(t);
  },[apiBase]);
  const startAB = async()=>{ const fd=new FormData(); fd.append('control',cfg.control); fd.append('shadow',cfg.shadow);
    fd.append('window_mins', String(cfg.windowMins)); fd.append('promote_sharpe_delta', String(cfg.promoteIf.sharpeDelta));
    fd.append('promote_dd_cap', String(cfg.promoteIf.ddCap)); await fetch(`${apiBase}/ab/start`,{method:'POST',body:fd});
    const s=await fetch(`${apiBase}/ab/status`).then(r=>r.json()); setStatus(s); setAbOn(true); };
  const stopAB = async()=>{ await fetch(`${apiBase}/ab/stop`,{method:'POST'}); const s=await fetch(`${apiBase}/ab/status`).then(r=>r.json()); setStatus(s); setAbOn(false); };
  const promote = async()=>{ await fetch(`${apiBase}/ab/promote_candidate`,{method:'POST'}); alert('Candidate promoted to control.'); };
  return <div className="space-y-3">
    <div className="flex flex-wrap items-center gap-2">
      <label className="text-xs text-gray-600">Control</label>
      <select className="border rounded px-2 py-1 text-xs" value={cfg.control} onChange={e=>setCfg({...cfg, control:e.target.value})}><option value="live">Live</option><option value="paper">Paper</option></select>
      <label className="text-xs text-gray-600">Shadow</label>
      <select className="border rounded px-2 py-1 text-xs" value={cfg.shadow} onChange={e=>setCfg({...cfg, shadow:e.target.value})}><option value="paper">Paper</option><option value="live">Live</option></select>
      <label className="text-xs text-gray-600">Window (min)</label>
      <input type="number" className="border rounded px-2 py-1 w-20" value={cfg.windowMins} onChange={e=>setCfg({...cfg, windowMins:Number(e.target.value)})}/>
      <label className="text-xs text-gray-600">SharpeΔ ≥</label>
      <input type="number" step="0.05" className="border rounded px-2 py-1 w-24" value={cfg.promoteIf.sharpeDelta} onChange={e=>setCfg({...cfg, promoteIf:{...cfg.promoteIf, sharpeDelta:Number(e.target.value)}})}/>
      <label className="text-xs text-gray-600">DD ≤</label>
      <input type="number" step="0.1" className="border rounded px-2 py-1 w-24" value={cfg.promoteIf.ddCap} onChange={e=>setCfg({...cfg, promoteIf:{...cfg.promoteIf, ddCap:Number(e.target.value)}})}/>
      {!abOn ? (<button onClick={startAB} className="px-3 py-2 rounded-lg bg-black text-white text-xs">Start A/B</button>)
              : (<button onClick={stopAB} className="px-3 py-2 rounded-lg bg-red-600 text-white text-xs">Stop A/B</button>)}
      <button onClick={promote} className="px-3 py-2 rounded-lg bg-green-600 text-white text-xs">Promote Candidate</button>
    </div>
    <div className="grid md:grid-cols-2 gap-3 text-xs">
      <div className="bg-gray-50 rounded-lg p-3 border"><div className="font-semibold mb-1">Control</div><pre className="whitespace-pre-wrap">{JSON.stringify(status.control||{},null,2)}</pre></div>
      <div className="bg-gray-50 rounded-lg p-3 border"><div className="font-semibold mb-1">Candidate</div><pre className="whitespace-pre-wrap">{JSON.stringify(status.candidate||{},null,2)}</pre></div>
    </div>
    <div className="text-xs text-gray-600">State: <b>{status.state||'idle'}</b> • Window ends: {status.windowEnds||'—'} • Eligible: {String(status.eligible??false)}</div>
  </div>
}

export default function PineLabUnifiedDashboard({ apiBase }: { apiBase: string }){
  const [summary, setSummary] = useState<any>({ equity: 0, cash: 0, unrealized: 0, ordersToday: 0, health: 'Good' });
  const [chartData, setChartData] = useState<any[]>([]);
  const [performanceMode, setPerformanceMode] = useState(false);
  const [metrics, setMetrics] = useState<any>({ sharpe: 0, drawdown: 0, corr: [], benchmarks: [] });
  const [autoTuneOn, setAutoTuneOn] = useState(false);
  const [autoTuneCfg, setAutoTuneCfg] = useState({ cadenceMin: 30, sharpeTarget: 1.0, maxDD: 10, trials: 50, sampler: 'tpe' });
  const [autoTuneStatus, setAutoTuneStatus] = useState<any>({ state: 'idle', lastRun: null, bestScore: null, bestParams: null });
  const [bestParams, setBestParams] = useState<any>(null);

  useEffect(()=>{
    const fetchSummary = async()=>{
      try{
        const pnl = await fetch(`${apiBase}/pnl/summary`).then(r=>r.json());
        const journal = await fetch(`${apiBase}/journal/orders`).then(r=>r.json());
        const today = journal.filter((j:any)=> j.time && j.time.startsWith(new Date().toISOString().slice(0,10))).length;
        const eq = pnl.equity||0, cash = pnl.cash||0, unreal = pnl.unrealized||0;
        const pnlPct = eq && cash ? ((eq-cash)/cash)*100 : 0;
        setSummary({ equity:eq, cash, unrealized:unreal, pnlPct, ordersToday: today, health: pnlPct > -1 ? 'Good':'Review' });
        const ts = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
        setChartData(prev=>[...prev.slice(-30), {time:ts, equity:eq, pnl:pnlPct}]);
      }catch{}
    };
    fetchSummary();
    const t = setInterval(fetchSummary, 20000);
    return ()=>clearInterval(t);
  },[apiBase]);

  useEffect(()=>{
    let t:any;
    const poll = async()=>{
      try{
        const s = await fetch(`${apiBase}/autotune/status`).then(r=>r.json());
        setAutoTuneStatus(s); setAutoTuneOn(!!s.enabled);
        if(s?.promoted?.params) setBestParams(s.promoted.params);
      }catch{} t=setTimeout(poll,15000);
    }; poll(); return ()=>clearTimeout(t);
  },[apiBase]);

  const startAutoTune = async()=>{ const fd=new FormData(); Object.entries(autoTuneCfg).forEach(([k,v])=>fd.append(k,String(v)));
    await fetch(`${apiBase}/autotune/start_bayes`,{method:'POST', body:fd});
    const s = await fetch(`${apiBase}/autotune/status`).then(r=>r.json()); setAutoTuneStatus(s); setAutoTuneOn(True)};

  const stopAutoTune = async()=>{ await fetch(`${apiBase}/autotune/stop`,{method:'POST'}); const s=await fetch(`${apiBase}/autotune/status`).then(r=>r.json()); setAutoTuneStatus(s); setAutoTuneOn(false)};

  const promoteBest = async()=>{ await fetch(`${apiBase}/autotune/promote_best`,{method:'POST'}); alert('Best parameters promoted to active strategy!'); };

  const useBestParams = async()=>{
    try{ const best = await fetch(`${apiBase}/strategy/params/best`).then(r=>r.json());
      if(!best?.params) return alert('No promoted parameters found.');
      setBestParams(best.params); alert('Loaded promoted best parameters into Strategy Studio.');
    }catch{ alert('Error loading best parameters.'); }
  };

  return <div className="p-6 space-y-6">
    <div className="grid md:grid-cols-5 gap-3">
      <MetricCard title="Equity" value={`$${summary.equity.toFixed(2)}`}/>
      <MetricCard title="Cash" value={`$${summary.cash.toFixed(2)}`}/>
      <MetricCard title="Unrealized P&L" value={`$${summary.unrealized.toFixed(2)}`} good={summary.unrealized>=0}/>
      <MetricCard title="Orders Today" value={summary.ordersToday}/>
      <MetricCard title="Strategy Health" value={summary.health} change={`${summary.pnlPct?.toFixed(2)}%`} good={summary.health==='Good'}/>
    </div>

    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold">Auto-Optimization & Walk-Forward Validation</h3>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-600">Cadence</label>
          <input type="number" className="border rounded px-2 py-1 w-16" value={autoTuneCfg.cadenceMin} onChange={e=>setAutoTuneCfg({...autoTuneCfg, cadenceMin:Number(e.target.value)})}/>
          <label className="text-xs text-gray-600">Trials</label>
          <input type="number" className="border rounded px-2 py-1 w-16" value={autoTuneCfg.trials} onChange={e=>setAutoTuneCfg({...autoTuneCfg, trials:Number(e.target.value)})}/>
          <label className="text-xs text-gray-600">Sampler</label>
          <select className="border rounded px-2 py-1 text-xs" value={autoTuneCfg.sampler} onChange={e=>setAutoTuneCfg({...autoTuneCfg, sampler:e.target.value})}>
            <option value="tpe">TPE</option><option value="cmaes">CMA-ES</option><option value="random">Random</option>
          </select>
          {!autoTuneOn ? (<button onClick={startAutoTune} className="px-3 py-2 rounded-lg bg-black text-white text-xs">Start</button>)
                        : (<button onClick={stopAutoTune} className="px-3 py-2 rounded-lg bg-red-600 text-white text-xs">Stop</button>)}
          <button onClick={promoteBest} className="px-3 py-2 rounded-lg bg-green-600 text-white text-xs">Promote Best</button>
          <button onClick={useBestParams} className="px-3 py-2 rounded-lg bg-blue-600 text-white text-xs">Use Best Params</button>
        </div>
      </div>
      <div className="grid md:grid-cols-3 gap-3 text-xs text-gray-700">
        <div className="bg-gray-50 rounded-lg p-2"><b>Status:</b> {autoTuneStatus.state||'idle'}</div>
        <div className="bg-gray-50 rounded-lg p-2"><b>Last Run:</b> {autoTuneStatus.lastRun||'—'}</div>
        <div className="bg-gray-50 rounded-lg p-2 overflow-auto"><b>Best:</b> {autoTuneStatus.bestScore?`score=${autoTuneStatus.bestScore}`:'—'} {autoTuneStatus.bestParams?`params=${JSON.stringify(autoTuneStatus.bestParams)}`:''}</div>
      </div>
    </div>

    {bestParams && (<div className="bg-green-50 border border-green-200 rounded-xl p-3 text-xs text-gray-800">
      <div className="font-semibold mb-1">Current Best Parameters in Studio:</div>
      <pre className="whitespace-pre-wrap">{JSON.stringify(bestParams,null,2)}</pre>
    </div>)}

    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="flex items-center justify-between mb-2"><h3 className="text-sm font-semibold">A/B Live Shadow Deployments</h3></div>
      <ABLiver apiBase={apiBase}/>
    </div>
  </div>
}
