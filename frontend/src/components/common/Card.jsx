import React from 'react'

export function Card({ title, value, subtitle, status, className = '', children }) {
  const statusColors = {
    success: 'text-green-600',
    warning: 'text-yellow-600',
    error: 'text-red-600',
    info: 'text-blue-600',
    default: 'text-gray-900',
  }

  const colorClass = statusColors[status] || statusColors.default

  return (
    <div className={`bg-white border rounded-xl shadow-sm p-4 ${className}`}>
      {title && <div className="text-xs text-gray-500 mb-1">{title}</div>}
      {value !== undefined && (
        <div className={`text-2xl font-bold ${colorClass}`}>{value}</div>
      )}
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
      {children}
    </div>
  )
}

export function StatCard({ label, value, change, trend }) {
  const trendColor = trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-600'
  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'

  return (
    <div className="bg-white border rounded-xl shadow-sm p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="flex items-baseline justify-between mt-1">
        <span className="text-2xl font-bold">{value}</span>
        {change !== undefined && (
          <span className={`text-sm font-semibold ${trendColor}`}>
            {trendIcon} {change}
          </span>
        )}
      </div>
    </div>
  )
}

export function MetricCard({ icon, label, value, color = 'blue' }) {
  const colors = {
    blue: 'bg-blue-50 text-blue-600 border-blue-200',
    green: 'bg-green-50 text-green-600 border-green-200',
    red: 'bg-red-50 text-red-600 border-red-200',
    yellow: 'bg-yellow-50 text-yellow-600 border-yellow-200',
    purple: 'bg-purple-50 text-purple-600 border-purple-200',
  }

  return (
    <div className={`rounded-xl border-2 p-4 ${colors[color] || colors.blue}`}>
      <div className="flex items-center gap-3">
        <div className="text-3xl">{icon}</div>
        <div className="flex-1">
          <div className="text-xs opacity-75">{label}</div>
          <div className="text-xl font-bold">{value}</div>
        </div>
      </div>
    </div>
  )
}
