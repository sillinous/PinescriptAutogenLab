// frontend/src/components/analytics/AnalyticsDashboard.jsx
/**
 * Analytics Dashboard Component
 *
 * Displays comprehensive trading analytics including:
 * - Portfolio summary and positions
 * - Trade metrics and performance
 * - Risk exposure analysis
 * - Strategy performance comparison
 */

import React, { useState, useEffect } from 'react';
import { analyticsAPI } from '../../services/api';
import { Card } from '../common/Card';
import { LoadingCard as Loading } from '../common/Loading';

// ============================================================================
// Sub-Components
// ============================================================================

function MetricCard({ title, value, subtitle, trend, className = '' }) {
  const trendColor = trend > 0 ? 'text-green-500' : trend < 0 ? 'text-red-500' : 'text-gray-500';
  const trendIcon = trend > 0 ? '↑' : trend < 0 ? '↓' : '→';

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <p className="text-gray-400 text-sm">{title}</p>
      <p className="text-2xl font-bold text-white mt-1">{value}</p>
      {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
      {trend !== undefined && (
        <p className={`text-sm mt-1 ${trendColor}`}>
          {trendIcon} {Math.abs(trend).toFixed(2)}%
        </p>
      )}
    </div>
  );
}

function PositionRow({ position }) {
  const pnlColor = position.unrealized_pnl >= 0 ? 'text-green-500' : 'text-red-500';

  return (
    <tr className="border-b border-gray-700">
      <td className="py-2 px-3 text-white font-medium">{position.symbol}</td>
      <td className="py-2 px-3 text-gray-300">{position.side}</td>
      <td className="py-2 px-3 text-gray-300">{position.qty}</td>
      <td className="py-2 px-3 text-gray-300">${position.avg_entry_price?.toFixed(2)}</td>
      <td className="py-2 px-3 text-gray-300">${position.current_price?.toFixed(2) || '-'}</td>
      <td className={`py-2 px-3 ${pnlColor}`}>
        ${position.unrealized_pnl?.toFixed(2) || '0.00'}
        {position.unrealized_pnl_pct && (
          <span className="text-xs ml-1">({position.unrealized_pnl_pct.toFixed(2)}%)</span>
        )}
      </td>
    </tr>
  );
}

function StrategyRow({ strategy }) {
  const pnlColor = strategy.total_pnl >= 0 ? 'text-green-500' : 'text-red-500';

  return (
    <tr className="border-b border-gray-700">
      <td className="py-2 px-3 text-white font-medium">{strategy.name}</td>
      <td className="py-2 px-3 text-gray-300">{strategy.trade_count}</td>
      <td className={`py-2 px-3 ${pnlColor}`}>${strategy.total_pnl.toFixed(2)}</td>
      <td className="py-2 px-3">
        <span className={`px-2 py-1 rounded text-xs ${
          strategy.status === 'active' ? 'bg-green-900 text-green-300' : 'bg-gray-700 text-gray-300'
        }`}>
          {strategy.status}
        </span>
      </td>
    </tr>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function AnalyticsDashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [portfolioSummary, setPortfolioSummary] = useState(null);
  const [positions, setPositions] = useState([]);
  const [tradeMetrics, setTradeMetrics] = useState(null);
  const [riskMetrics, setRiskMetrics] = useState(null);
  const [riskExposure, setRiskExposure] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [selectedPeriod, setSelectedPeriod] = useState('month');

  useEffect(() => {
    fetchAllData();
  }, [selectedPeriod]);

  async function fetchAllData() {
    setLoading(true);
    setError(null);

    try {
      const [
        summaryRes,
        positionsRes,
        metricsRes,
        riskRes,
        exposureRes,
        strategiesRes
      ] = await Promise.allSettled([
        analyticsAPI.getPortfolioSummary(),
        analyticsAPI.getPortfolioPositions(),
        analyticsAPI.getTradeMetrics(selectedPeriod),
        analyticsAPI.getRiskMetrics(),
        analyticsAPI.getRiskExposure(),
        analyticsAPI.listStrategies()
      ]);

      if (summaryRes.status === 'fulfilled') setPortfolioSummary(summaryRes.value);
      if (positionsRes.status === 'fulfilled') setPositions(positionsRes.value);
      if (metricsRes.status === 'fulfilled') setTradeMetrics(metricsRes.value);
      if (riskRes.status === 'fulfilled') setRiskMetrics(riskRes.value);
      if (exposureRes.status === 'fulfilled') setRiskExposure(exposureRes.value);
      if (strategiesRes.status === 'fulfilled') setStrategies(strategiesRes.value);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <Loading message="Loading analytics..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Trading Analytics</h2>
        <div className="flex gap-2">
          {['day', 'week', 'month', 'quarter', 'year', 'all'].map(period => (
            <button
              key={period}
              onClick={() => setSelectedPeriod(period)}
              className={`px-3 py-1 rounded text-sm ${
                selectedPeriod === period
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {period.charAt(0).toUpperCase() + period.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Portfolio Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          title="Total Equity"
          value={`$${portfolioSummary?.total_equity?.toLocaleString() || '0'}`}
        />
        <MetricCard
          title="Cash Available"
          value={`$${portfolioSummary?.total_cash?.toLocaleString() || '0'}`}
        />
        <MetricCard
          title="Positions Value"
          value={`$${portfolioSummary?.total_positions_value?.toLocaleString() || '0'}`}
          subtitle={`${portfolioSummary?.position_count || 0} positions`}
        />
        <MetricCard
          title="Unrealized P&L"
          value={`$${portfolioSummary?.total_unrealized_pnl?.toFixed(2) || '0'}`}
          trend={portfolioSummary?.total_unrealized_pnl}
        />
      </div>

      {/* Trade Metrics */}
      <Card title="Trade Metrics" subtitle={`Period: ${selectedPeriod}`}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="Total Trades"
            value={tradeMetrics?.total_trades || 0}
          />
          <MetricCard
            title="Win Rate"
            value={`${((tradeMetrics?.win_rate || 0) * 100).toFixed(1)}%`}
          />
          <MetricCard
            title="Realized P&L"
            value={`$${tradeMetrics?.realized_pnl?.toFixed(2) || '0'}`}
            trend={tradeMetrics?.realized_pnl}
          />
          <MetricCard
            title="Profit Factor"
            value={tradeMetrics?.profit_factor?.toFixed(2) || '-'}
          />
        </div>
      </Card>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Risk Metrics">
          <div className="grid grid-cols-2 gap-4">
            <MetricCard
              title="Total Exposure"
              value={`$${riskMetrics?.total_exposure?.toLocaleString() || '0'}`}
            />
            <MetricCard
              title="Top 5 Concentration"
              value={`${riskMetrics?.top_5_concentration?.toFixed(1) || 0}%`}
            />
            <MetricCard
              title="Diversification"
              value={`${((riskMetrics?.diversification_score || 0) * 100).toFixed(1)}%`}
            />
            <MetricCard
              title="Volatility (Ann.)"
              value={riskMetrics?.annualized_volatility ? `${riskMetrics.annualized_volatility}%` : '-'}
            />
          </div>
        </Card>

        <Card title="Exposure Breakdown">
          <div className="grid grid-cols-2 gap-4">
            <MetricCard
              title="Long Exposure"
              value={`$${riskExposure?.long_exposure?.toLocaleString() || '0'}`}
              className="bg-green-900/30"
            />
            <MetricCard
              title="Short Exposure"
              value={`$${riskExposure?.short_exposure?.toLocaleString() || '0'}`}
              className="bg-red-900/30"
            />
            <MetricCard
              title="Net Exposure"
              value={`$${riskExposure?.net_exposure?.toLocaleString() || '0'}`}
            />
            <MetricCard
              title="L/S Ratio"
              value={riskExposure?.long_short_ratio?.toFixed(2) || '-'}
            />
          </div>
        </Card>
      </div>

      {/* Positions Table */}
      <Card title="Open Positions" subtitle={`${positions.length} positions`}>
        {positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="py-2 px-3">Symbol</th>
                  <th className="py-2 px-3">Side</th>
                  <th className="py-2 px-3">Qty</th>
                  <th className="py-2 px-3">Entry</th>
                  <th className="py-2 px-3">Current</th>
                  <th className="py-2 px-3">P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, idx) => (
                  <PositionRow key={idx} position={pos} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No open positions</p>
        )}
      </Card>

      {/* Strategies */}
      <Card title="Strategy Performance" subtitle={`${strategies.length} strategies`}>
        {strategies.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                  <th className="py-2 px-3">Strategy</th>
                  <th className="py-2 px-3">Trades</th>
                  <th className="py-2 px-3">Total P&L</th>
                  <th className="py-2 px-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {strategies.map((strategy, idx) => (
                  <StrategyRow key={idx} strategy={strategy} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No strategies found</p>
        )}
      </Card>

      {/* Refresh Button */}
      <div className="flex justify-center">
        <button
          onClick={fetchAllData}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition"
        >
          Refresh Data
        </button>
      </div>
    </div>
  );
}
