import React, { useEffect, useMemo, useRef, useState } from "react";

/* ---------------- Utilities ---------------- */

function safeText(v: any, fallback: string = "—"): string {
  if (v === undefined || v === null) return fallback;
  const t = typeof v;
  if (t === "string" || t === "number" || t === "boolean") return String(v);
  try {
    return JSON.stringify(v);
  } catch {
    return fallback;
  }
}

// simple polling hook with cleanup
function usePoll<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  intervalMs: number,
  deps: any[] = [],
  { immediate = true }: { immediate?: boolean } = {}
) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<Error | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    const run = async () => {
      try {
        const res = await fn(controller.signal);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted && e?.name !== "AbortError") setErr(e);
      }
    };

    if (immediate) run();
    timerRef.current = window.setInterval(run, Math.max(1000, intervalMs));

    return () => {
      mounted = false;
      controller.abort();
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, err } as const;
}

/* ---------------- Small UI bits ---------------- */

function Card({
  title,
  value,
  hint,
  good,
}: {
  title: string;
  value: React.ReactNode;
  hint?: string;
  good?: boolean;
}) {
  return (
    <div className="flex flex-col p-4 bg-white rounded-xl border shadow-sm">
      <div className="text-xs text-gray-500">{title}</div>
      <div className={`text-xl font-semibold ${good === false ? "text-red-600" : ""}`}>
        {value}
      </div>
      {hint ? (
        <div className={`text-[11px] ${good ? "text-green-600" : "text-gray-500"} mt-0.5`}>
          {hint}
        </div>
      ) : null}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function Progress({ pct }: { pct: number }) {
  const clamped = Math.max(0, Math.min(100, pct || 0));
  return (
    <div className="w-full h-2 rounded-full bg-gray-200 overflow-hidden">
      <div
        className="h-2 bg-green-600"
        style={{ width: `${clamped}%`, transition: "width .4s ease" }}
      />
    </div>
  );
}

/* ---------------- Main Dashboard ---------------- */

export default function PineLabUnifiedDashboard({
  apiBase = "http://localhost:8080",
}: {
  apiBase?: string;
}) {
  // PnL & journal
  const { data: pnlData } = usePoll(
    async (signal) => {
      const [pnl, journal] = await Promise.all([
        fetch(`${apiBase}/pnl/summary`, { signal }).then((r) => r.json()),
        fetch(`${apiBase}/journal/orders`, { signal }).then((r) => r.json()),
      ]);
      const todayCount = Array.isArray(journal)
        ? journal.filter(
            (j: any) =>
              j?.id && j?.symbol && String(j?.time || "").startsWith(new Date().toISOString().slice(0, 10))
          ).length
        : 0;
      return { pnl, todayCount };
    },
    15000,
    [apiBase]
  );

  // Auto-tune status
  const { data: tuneData } = usePoll(
    async (signal) => {
      const s = await fetch(`${apiBase}/autotune/status`, { signal }).then((r) => r.json());
      return s;
    },
    12000,
    [apiBase]
  );

  // AB status
  const { data: abData } = usePoll(
    async (signal) => {
      const s = await fetch(`${apiBase}/ab/status`, { signal }).then((r) => r.json());
      return s;
    },
    12000,
    [apiBase]
  );

  const equity$ = useMemo(() => safeText(pnlData?.pnl?.net_profit ?? 0), [pnlData]);
  const trades = useMemo(() => Number(pnlData?.pnl?.total_trades ?? 0), [pnlData]);
  const winRate = useMemo(() => Number(pnlData?.pnl?.win_rate ?? 0), [pnlData]);
  const pf = useMemo(() => Number(pnlData?.pnl?.profit_factor ?? 0), [pnlData]);
  const ordersToday = useMemo(() => Number(pnlData?.todayCount ?? 0), [pnlData]);

  const tunePct = useMemo(() => Number(tuneData?.progress ?? 0), [tuneData]);
  const tuneBest = useMemo(() => tuneData?.best_parameters ?? {}, [tuneData]);

  const aWin = useMemo(() => Number(abData?.variant_a_winrate ?? 0), [abData]);
  const bWin = useMemo(() => Number(abData?.variant_b_winrate ?? 0), [abData]);
  const abWinner = useMemo(() => safeText(abData?.winner ?? "—"), [abData]);
  const abName = useMemo(() => safeText(abData?.test_name ?? "—"), [abData]);

  // Actions (mock back to your backend)
  const handleStartAB = async () => {
    // In your real backend, add POST /ab/start to control a test window
    alert("A/B start requested (wire POST /ab/start in backend when ready)");
  };
  const handlePromote = async () => {
    alert("Promote requested (wire POST /ab/promote_candidate in backend when ready)");
  };

  return (
    <div className="p-6 space-y-6">
      {/* Top cards */}
      <div className="grid gap-3 md:grid-cols-5">
        <Card title="Net P&L ($)" value={`$${equity$}`} />
        <Card title="Total Trades" value={trades} />
        <Card title="Win Rate" value={`${winRate.toFixed(2)}%`} />
        <Card title="Profit Factor" value={pf.toFixed(2)} good={pf >= 1.2} />
        <Card title="Orders Today" value={ordersToday} />
      </div>

      {/* Auto-Optimization */}
      <Section title="Auto-Optimization">
        <div className="grid gap-3 md:grid-cols-3">
          <div className="p-3 rounded-lg border bg-gray-50">
            <div className="text-xs text-gray-600 mb-1">Progress</div>
            <Progress pct={isFinite(tunePct) ? tunePct : 0} />
            <div className="text-[11px] mt-1 text-gray-500">{tunePct.toFixed(1)}%</div>
          </div>
          <div className="p-3 rounded-lg border bg-gray-50 md:col-span-2">
            <div className="text-xs text-gray-600 mb-1">Best Parameters</div>
            <pre className="text-[11px] whitespace-pre-wrap">
              {safeText(tuneBest, "{}")}
            </pre>
          </div>
        </div>
      </Section>

      {/* A/B Testing */}
      <Section title="A/B Live Shadow Deployments">
        <div className="grid gap-3 md:grid-cols-3">
          <div className="p-3 rounded-lg border bg-gray-50">
            <div className="text-xs text-gray-600 mb-1">Test</div>
            <div className="text-sm font-semibold">{abName}</div>
            <div className="text-[11px] mt-1 text-gray-500">
              Winner: <span className="font-medium">{abWinner}</span>
            </div>
          </div>
          <div className="p-3 rounded-lg border bg-gray-50">
            <div className="text-xs text-gray-600 mb-1">Variant A Winrate</div>
            <div className="text-sm font-semibold">{isFinite(aWin) ? `${aWin.toFixed(2)}%` : "—"}</div>
          </div>
          <div className="p-3 rounded-lg border bg-gray-50">
            <div className="text-xs text-gray-600 mb-1">Variant B Winrate</div>
            <div className="text-sm font-semibold">{isFinite(bWin) ? `${bWin.toFixed(2)}%` : "—"}</div>
          </div>
        </div>
        <div className="flex gap-2 mt-3">
          <button
            onClick={handleStartAB}
            className="px-3 py-2 rounded-lg bg-black text-white text-xs"
          >
            Start A/B
          </button>
          <button
            onClick={handlePromote}
            className="px-3 py-2 rounded-lg bg-green-600 text-white text-xs"
          >
            Promote Candidate
          </button>
        </div>
      </Section>
    </div>
  );
}
