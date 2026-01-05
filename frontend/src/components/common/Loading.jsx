import React from 'react'

export function LoadingSpinner({ size = 'md', className = '' }) {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
  }

  return (
    <div className={`inline-block ${className}`}>
      <div
        className={`animate-spin rounded-full border-b-2 border-blue-600 ${sizes[size]}`}
      />
    </div>
  )
}

export function LoadingCard({ message = 'Loading...' }) {
  return (
    <div className="bg-white border rounded-xl shadow-sm p-8 text-center">
      <LoadingSpinner size="lg" className="mb-4" />
      <div className="text-sm text-gray-600">{message}</div>
    </div>
  )
}

export function LoadingOverlay({ message = 'Loading...' }) {
  return (
    <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-50 rounded-xl">
      <div className="text-center">
        <LoadingSpinner size="lg" className="mb-4" />
        <div className="text-sm text-gray-600">{message}</div>
      </div>
    </div>
  )
}
