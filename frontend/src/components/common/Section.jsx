import React from 'react'

export function Section({ title, badge, action, children, className = '' }) {
  return (
    <div className={`bg-white border rounded-xl shadow-sm p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          {title}
          {badge && (
            <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
              {badge}
            </span>
          )}
        </h3>
        {action}
      </div>
      {children}
    </div>
  )
}

export function SectionHeader({ title, description, action }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">{title}</h2>
        {description && <p className="text-sm text-gray-600 mt-1">{description}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}
