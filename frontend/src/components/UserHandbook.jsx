import React, { useState } from 'react'
import { Card } from './common/Card'

export default function UserHandbook() {
  const [activeSection, setActiveSection] = useState('getting-started')
  const [searchTerm, setSearchTerm] = useState('')

  const sections = [
    {
      id: 'getting-started',
      icon: '🚀',
      title: 'Getting Started',
      subsections: [
        { id: 'intro', title: 'Introduction' },
        { id: 'quick-start', title: 'Quick Start Guide' },
        { id: 'navigation', title: 'Platform Navigation' },
      ],
    },
    {
      id: 'overview',
      icon: '📊',
      title: 'Overview Dashboard',
      subsections: [
        { id: 'signals', title: 'Signal Aggregator' },
        { id: 'charts', title: 'Price Charts' },
        { id: 'predictions', title: 'AI Predictions' },
      ],
    },
    {
      id: 'models',
      icon: '🎓',
      title: 'Model Lab',
      subsections: [
        { id: 'model-types', title: 'Model Types' },
        { id: 'training', title: 'Training Models' },
        { id: 'evaluation', title: 'Model Evaluation' },
      ],
    },
    {
      id: 'deep-learning',
      icon: '🧠',
      title: 'Deep Learning',
      subsections: [
        { id: 'lstm', title: 'LSTM Models' },
        { id: 'transformers', title: 'Transformers' },
        { id: 'ensembles', title: 'Ensemble Models' },
      ],
    },
    {
      id: 'features',
      icon: '🔬',
      title: 'Feature Engineering',
      subsections: [
        { id: 'feature-types', title: 'Feature Types' },
        { id: 'feature-explorer', title: 'Feature Explorer' },
      ],
    },
    {
      id: 'platform',
      icon: '⚙️',
      title: 'Platform Tools',
      subsections: [
        { id: 'ab-testing', title: 'A/B Testing' },
        { id: 'optimization', title: 'Auto Optimization' },
      ],
    },
    {
      id: 'workflows',
      icon: '🔄',
      title: 'Common Workflows',
      subsections: [
        { id: 'workflow-beginner', title: 'Beginner Workflow' },
        { id: 'workflow-advanced', title: 'Advanced Trading' },
        { id: 'workflow-research', title: 'Research & Analysis' },
      ],
    },
    {
      id: 'troubleshooting',
      icon: '🔧',
      title: 'Troubleshooting',
      subsections: [
        { id: 'common-issues', title: 'Common Issues' },
        { id: 'faq', title: 'FAQ' },
      ],
    },
  ]

  const content = {
    intro: {
      title: 'Welcome to PineLab AI Trading Platform',
      content: (
        <div className="space-y-4">
          <p className="text-lg text-gray-700">
            PineLab is a comprehensive AI-powered trading platform that combines cutting-edge machine learning
            with real-time cryptocurrency market data to provide intelligent trading insights.
          </p>

          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <h4 className="font-semibold text-blue-900 mb-2">🎯 Platform Capabilities</h4>
            <ul className="space-y-2 text-blue-800">
              <li>✓ Real-time market data from 900+ cryptocurrency pairs</li>
              <li>✓ Advanced AI signal aggregation from multiple sources</li>
              <li>✓ Deep learning models (LSTM, Transformers, Ensembles)</li>
              <li>✓ Automated feature engineering and selection</li>
              <li>✓ A/B testing and strategy optimization</li>
              <li>✓ Support/Resistance level detection</li>
            </ul>
          </div>

          <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
            <h4 className="font-semibold text-green-900 mb-2">💡 What You Can Do</h4>
            <ul className="space-y-2 text-green-800">
              <li><strong>Monitor Markets:</strong> Track live price data and trends across multiple timeframes</li>
              <li><strong>AI Predictions:</strong> Get ML-powered price predictions and trading signals</li>
              <li><strong>Train Models:</strong> Create custom AI models tailored to specific trading pairs</li>
              <li><strong>Backtest Strategies:</strong> Test trading strategies on historical data</li>
              <li><strong>Optimize Performance:</strong> Use auto-tuning to improve model accuracy</li>
            </ul>
          </div>
        </div>
      ),
    },
    'quick-start': {
      title: 'Quick Start Guide',
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-6">
            <h3 className="text-xl font-bold text-purple-900 mb-4">🏁 5-Minute Quick Start</h3>

            <div className="space-y-4">
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <div className="flex items-start">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold mr-3 flex-shrink-0">1</div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">Select Your Trading Pair</h4>
                    <p className="text-gray-600 text-sm">Use the Symbol dropdown in the header to choose from 900+ cryptocurrencies (e.g., BTC_USDT, ETH_USDT)</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 shadow-sm">
                <div className="flex items-start">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold mr-3 flex-shrink-0">2</div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">Choose Your Timeframe</h4>
                    <p className="text-gray-600 text-sm">Select a timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d) to match your trading strategy</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 shadow-sm">
                <div className="flex items-start">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold mr-3 flex-shrink-0">3</div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">Review AI Signals</h4>
                    <p className="text-gray-600 text-sm">Check the Signal Aggregator card for BUY/SELL/HOLD recommendations with confidence scores</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 shadow-sm">
                <div className="flex items-start">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold mr-3 flex-shrink-0">4</div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">Analyze Price Action</h4>
                    <p className="text-gray-600 text-sm">Study the price chart with support/resistance levels and volume indicators</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 shadow-sm">
                <div className="flex items-start">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold mr-3 flex-shrink-0">5</div>
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-1">Explore Advanced Features</h4>
                    <p className="text-gray-600 text-sm">Navigate to Model Lab or Deep Learning tabs to train custom AI models</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
            <h4 className="font-semibold text-yellow-900 mb-2">⚠️ Important Notes</h4>
            <ul className="space-y-1 text-yellow-800 text-sm">
              <li>• AI predictions are probabilistic - always use risk management</li>
              <li>• Markets are updated every 10 seconds in real-time</li>
              <li>• Model training requires at least 100 candles of historical data</li>
              <li>• Higher timeframes (4h, 1d) are generally more reliable for predictions</li>
            </ul>
          </div>
        </div>
      ),
    },
    navigation: {
      title: 'Platform Navigation',
      content: (
        <div className="space-y-6">
          <p className="text-gray-700">The platform is organized into five main tabs, each serving a specific purpose:</p>

          <div className="grid gap-4">
            <div className="bg-white border-2 border-blue-200 rounded-lg p-5 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">📊</span>
                <h3 className="text-xl font-bold text-gray-900">Overview Dashboard</h3>
              </div>
              <p className="text-gray-600 mb-3">Your main trading hub with real-time data and AI signals</p>
              <ul className="text-sm text-gray-700 space-y-1 ml-4">
                <li>• Live price charts with technical indicators</li>
                <li>• Aggregated AI trading signals (BUY/SELL/HOLD)</li>
                <li>• Price predictions panel</li>
                <li>• System health metrics</li>
              </ul>
            </div>

            <div className="bg-white border-2 border-purple-200 rounded-lg p-5 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🎓</span>
                <h3 className="text-xl font-bold text-gray-900">Model Lab</h3>
              </div>
              <p className="text-gray-600 mb-3">Manage and train AI models for trading strategies</p>
              <ul className="text-sm text-gray-700 space-y-1 ml-4">
                <li>• Create new ML models</li>
                <li>• Train models on historical data</li>
                <li>• Evaluate model performance</li>
                <li>• Deploy models for live predictions</li>
              </ul>
            </div>

            <div className="bg-white border-2 border-green-200 rounded-lg p-5 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🧠</span>
                <h3 className="text-xl font-bold text-gray-900">Deep Learning</h3>
              </div>
              <p className="text-gray-600 mb-3">Advanced neural network models for price forecasting</p>
              <ul className="text-sm text-gray-700 space-y-1 ml-4">
                <li>• LSTM (Long Short-Term Memory) price predictors</li>
                <li>• Transformer-based sequence models</li>
                <li>• Ensemble model aggregation</li>
                <li>• Prediction visualization and analysis</li>
              </ul>
            </div>

            <div className="bg-white border-2 border-orange-200 rounded-lg p-5 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🔬</span>
                <h3 className="text-xl font-bold text-gray-900">Features Explorer</h3>
              </div>
              <p className="text-gray-600 mb-3">Engineered features for advanced market analysis</p>
              <ul className="text-sm text-gray-700 space-y-1 ml-4">
                <li>• Technical indicators (RSI, MACD, Bollinger Bands)</li>
                <li>• Price momentum and volatility metrics</li>
                <li>• Volume-based features</li>
                <li>• Custom feature engineering</li>
              </ul>
            </div>

            <div className="bg-white border-2 border-gray-200 rounded-lg p-5 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">⚙️</span>
                <h3 className="text-xl font-bold text-gray-900">Platform Tools</h3>
              </div>
              <p className="text-gray-600 mb-3">Optimization and experimentation tools</p>
              <ul className="text-sm text-gray-700 space-y-1 ml-4">
                <li>• A/B testing for strategy comparison</li>
                <li>• Auto-optimization with hyperparameter tuning</li>
                <li>• Performance metrics and analytics</li>
                <li>• System configuration</li>
              </ul>
            </div>
          </div>
        </div>
      ),
    },
    signals: {
      title: 'Signal Aggregator',
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 p-5 rounded">
            <h3 className="text-lg font-bold text-blue-900 mb-2">What is the Signal Aggregator?</h3>
            <p className="text-blue-800">
              The Signal Aggregator combines multiple AI sources to provide a unified trading recommendation.
              It analyzes TradingView signals, AI model predictions, and reinforcement learning agents to
              generate a consensus signal with confidence scoring.
            </p>
          </div>

          <div className="space-y-4">
            <h4 className="font-semibold text-gray-900">📍 Signal Types</h4>

            <div className="grid gap-3">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <span className="text-2xl mr-2">🟢</span>
                  <h5 className="font-bold text-green-900">BUY Signal</h5>
                </div>
                <p className="text-green-800 text-sm">
                  Multiple AI sources agree on upward price movement. Higher confidence = stronger buy recommendation.
                </p>
              </div>

              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <span className="text-2xl mr-2">🔴</span>
                  <h5 className="font-bold text-red-900">SELL Signal</h5>
                </div>
                <p className="text-red-800 text-sm">
                  AI models predict downward price movement. Higher confidence = stronger sell recommendation.
                </p>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <span className="text-2xl mr-2">⚪</span>
                  <h5 className="font-bold text-gray-900">HOLD Signal</h5>
                </div>
                <p className="text-gray-800 text-sm">
                  Mixed signals or low confidence. Wait for clearer market direction before taking action.
                </p>
              </div>
            </div>

            <h4 className="font-semibold text-gray-900 mt-6">🎯 How to Use Signal Aggregator</h4>
            <ol className="space-y-3 ml-4">
              <li className="flex items-start">
                <span className="font-bold text-blue-600 mr-2">1.</span>
                <div>
                  <strong>Check Confidence Score:</strong> Look for signals with &gt;70% confidence for reliable trades
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-blue-600 mr-2">2.</span>
                <div>
                  <strong>Review Source Agreement:</strong> Signals are stronger when all sources (TradingView, AI, RL) agree
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-blue-600 mr-2">3.</span>
                <div>
                  <strong>Verify with Charts:</strong> Always confirm signals against price action and support/resistance levels
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-blue-600 mr-2">4.</span>
                <div>
                  <strong>Consider Position Size:</strong> Use the recommended position size based on signal strength
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-blue-600 mr-2">5.</span>
                <div>
                  <strong>Monitor Reasoning:</strong> Expand the reasoning section to understand why the signal was generated
                </div>
              </li>
            </ol>
          </div>

          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
            <h4 className="font-semibold text-yellow-900 mb-2">⚠️ Best Practices</h4>
            <ul className="space-y-1 text-yellow-800 text-sm">
              <li>• Never trade solely on AI signals - use them as one input in your decision-making</li>
              <li>• Combine signals with your own technical analysis and market knowledge</li>
              <li>• Lower timeframes (&lt;1h) produce more signals but may be less reliable</li>
              <li>• Always use stop-losses regardless of signal confidence</li>
              <li>• Backtest signals on historical data before live trading</li>
            </ul>
          </div>
        </div>
      ),
    },
    charts: {
      title: 'Advanced Price Charts',
      content: (
        <div className="space-y-6">
          <p className="text-gray-700">
            The Advanced Price Chart provides comprehensive technical analysis visualization with real-time
            market data, support/resistance levels, and volume indicators.
          </p>

          <div className="bg-blue-50 border-l-4 border-blue-500 p-5 rounded">
            <h4 className="font-semibold text-blue-900 mb-3">📊 Chart Features</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-semibold text-blue-800 mb-2">Price Visualization</h5>
                <ul className="text-sm text-blue-700 space-y-1">
                  <li>✓ Candlestick price data</li>
                  <li>✓ Real-time updates (10s intervals)</li>
                  <li>✓ Multiple timeframes (1m to 1d)</li>
                  <li>✓ Zoom and pan capabilities</li>
                </ul>
              </div>
              <div>
                <h5 className="font-semibold text-blue-800 mb-2">Technical Indicators</h5>
                <ul className="text-sm text-blue-700 space-y-1">
                  <li>✓ Support/Resistance levels</li>
                  <li>✓ Volume bars</li>
                  <li>✓ Moving averages</li>
                  <li>✓ Price trend lines</li>
                </ul>
              </div>
            </div>
          </div>

          <h4 className="font-semibold text-gray-900">🎨 Understanding Chart Elements</h4>

          <div className="space-y-3">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h5 className="font-bold text-gray-900 mb-2">🕯️ Candlesticks</h5>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-green-600">Green Candle (Bullish):</strong>
                  <p className="text-gray-700">Close price &gt; Open price. Buyers are in control.</p>
                </div>
                <div>
                  <strong className="text-red-600">Red Candle (Bearish):</strong>
                  <p className="text-gray-700">Close price &lt; Open price. Sellers are in control.</p>
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h5 className="font-bold text-gray-900 mb-2">📏 Support & Resistance</h5>
              <p className="text-gray-700 text-sm mb-2">
                Horizontal lines showing key price levels where the market has historically reversed:
              </p>
              <ul className="text-sm text-gray-700 space-y-1 ml-4">
                <li><strong className="text-green-600">Support (Green):</strong> Price floor - buyers step in</li>
                <li><strong className="text-red-600">Resistance (Red):</strong> Price ceiling - sellers dominate</li>
                <li><strong>Trading Strategy:</strong> Buy near support, sell near resistance</li>
              </ul>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h5 className="font-bold text-gray-900 mb-2">📊 Volume Bars</h5>
              <p className="text-gray-700 text-sm">
                Bar chart at bottom showing trading volume. Higher volume = stronger price movements.
                Look for volume confirmation when price breaks support/resistance levels.
              </p>
            </div>
          </div>

          <h4 className="font-semibold text-gray-900 mt-6">🔍 Chart Analysis Workflow</h4>
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-5">
            <ol className="space-y-3">
              <li className="flex items-start">
                <span className="font-bold text-purple-600 mr-2 min-w-[24px]">1.</span>
                <div className="text-sm">
                  <strong>Identify Trend:</strong> Look at overall price direction. Is it moving up, down, or sideways?
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-purple-600 mr-2 min-w-[24px]">2.</span>
                <div className="text-sm">
                  <strong>Locate Key Levels:</strong> Find support/resistance zones where price has reversed multiple times
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-purple-600 mr-2 min-w-[24px]">3.</span>
                <div className="text-sm">
                  <strong>Check Volume:</strong> Confirm price movements with volume - high volume = more reliable moves
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-purple-600 mr-2 min-w-[24px]">4.</span>
                <div className="text-sm">
                  <strong>Look for Patterns:</strong> Identify chart patterns (triangles, head & shoulders, double tops/bottoms)
                </div>
              </li>
              <li className="flex items-start">
                <span className="font-bold text-purple-600 mr-2 min-w-[24px]">5.</span>
                <div className="text-sm">
                  <strong>Combine with AI:</strong> Cross-reference chart analysis with AI signal predictions
                </div>
              </li>
            </ol>
          </div>
        </div>
      ),
    },
    predictions: {
      title: 'AI Predictions Panel',
      content: (
        <div className="space-y-6">
          <p className="text-gray-700">
            The AI Predictions Panel displays machine learning model forecasts for future price movements,
            including predicted price targets, confidence intervals, and time horizons.
          </p>

          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border-l-4 border-indigo-500 p-5 rounded">
            <h4 className="font-semibold text-indigo-900 mb-3">🔮 Prediction Components</h4>
            <div className="space-y-3 text-indigo-800">
              <div>
                <strong>Price Target:</strong> Expected price level within the forecast period
              </div>
              <div>
                <strong>Confidence Score:</strong> Model certainty (0-100%). Higher = more reliable
              </div>
              <div>
                <strong>Time Horizon:</strong> How far into the future the prediction extends
              </div>
              <div>
                <strong>Direction:</strong> UP (bullish), DOWN (bearish), or NEUTRAL
              </div>
            </div>
          </div>

          <h4 className="font-semibold text-gray-900">📈 Interpreting Predictions</h4>

          <div className="grid gap-3">
            <div className="bg-white border-l-4 border-green-500 p-4 rounded">
              <h5 className="font-bold text-green-900 mb-2">High Confidence (&gt;80%)</h5>
              <p className="text-sm text-gray-700">
                Model has strong conviction. Historical patterns closely match current conditions.
                These predictions are more reliable but should still be verified with technical analysis.
              </p>
            </div>

            <div className="bg-white border-l-4 border-yellow-500 p-4 rounded">
              <h5 className="font-bold text-yellow-900 mb-2">Medium Confidence (50-80%)</h5>
              <p className="text-sm text-gray-700">
                Model sees some patterns but uncertainty exists. Use as supplementary information
                alongside other analysis tools. Wait for additional confirmation.
              </p>
            </div>

            <div className="bg-white border-l-4 border-red-500 p-4 rounded">
              <h5 className="font-bold text-red-900 mb-2">Low Confidence (&lt;50%)</h5>
              <p className="text-sm text-gray-700">
                High uncertainty. Market conditions are unclear or unprecedented. Avoid making
                trading decisions based solely on these predictions.
              </p>
            </div>
          </div>

          <h4 className="font-semibold text-gray-900 mt-6">⚡ Quick Usage Guide</h4>
          <div className="bg-gray-50 rounded-lg p-5">
            <ol className="space-y-2 text-sm text-gray-700">
              <li><strong>1.</strong> Select your trading symbol from the header dropdown</li>
              <li><strong>2.</strong> Navigate to Overview tab to see the Predictions Panel</li>
              <li><strong>3.</strong> Review the predicted price target and direction</li>
              <li><strong>4.</strong> Check the confidence score - aim for &gt;70% for actionable insights</li>
              <li><strong>5.</strong> Note the time horizon to align with your trading timeframe</li>
              <li><strong>6.</strong> Verify prediction against current chart price action</li>
              <li><strong>7.</strong> Cross-reference with Signal Aggregator for confirmation</li>
            </ol>
          </div>

          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <h4 className="font-semibold text-blue-900 mb-2">💡 Pro Tips</h4>
            <ul className="space-y-1 text-blue-800 text-sm">
              <li>• Predictions are most accurate 1-4 candles into the future</li>
              <li>• Longer timeframes (4h, 1d) produce more reliable predictions</li>
              <li>• Always use predictions in combination with risk management</li>
              <li>• Model accuracy varies by market conditions - track performance over time</li>
              <li>• Retrain models periodically to adapt to changing market dynamics</li>
            </ul>
          </div>
        </div>
      ),
    },
    'model-types': {
      title: 'AI Model Types',
      content: (
        <div className="space-y-6">
          <p className="text-gray-700">
            PineLab supports multiple types of machine learning models, each with unique strengths
            for different market conditions and trading strategies.
          </p>

          <div className="grid gap-4">
            <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-300 rounded-lg p-5">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🧠</span>
                <h4 className="text-xl font-bold text-blue-900">LSTM Models</h4>
              </div>
              <p className="text-gray-700 mb-3">
                Long Short-Term Memory networks excel at learning patterns in sequential data.
                Perfect for capturing long-term price trends and recurring patterns.
              </p>
              <div className="bg-white rounded p-3 text-sm">
                <div className="font-semibold text-gray-900 mb-2">Best For:</div>
                <ul className="text-gray-700 space-y-1">
                  <li>✓ Trend following strategies</li>
                  <li>✓ Multi-day price forecasting</li>
                  <li>✓ Markets with clear momentum</li>
                  <li>✓ Longer timeframes (1h, 4h, 1d)</li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-300 rounded-lg p-5">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🔮</span>
                <h4 className="text-xl font-bold text-purple-900">Transformer Models</h4>
              </div>
              <p className="text-gray-700 mb-3">
                Attention-based models that can focus on the most relevant historical periods.
                Excellent for complex pattern recognition and multi-factor analysis.
              </p>
              <div className="bg-white rounded p-3 text-sm">
                <div className="font-semibold text-gray-900 mb-2">Best For:</div>
                <ul className="text-gray-700 space-y-1">
                  <li>✓ Complex market patterns</li>
                  <li>✓ Multi-indicator analysis</li>
                  <li>✓ Volatile markets</li>
                  <li>✓ Short to medium-term predictions</li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 rounded-lg p-5">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🎯</span>
                <h4 className="text-xl font-bold text-green-900">Ensemble Models</h4>
              </div>
              <p className="text-gray-700 mb-3">
                Combines multiple models (LSTM + Transformer) using weighted voting.
                Provides robust predictions by leveraging strengths of different architectures.
              </p>
              <div className="bg-white rounded p-3 text-sm">
                <div className="font-semibold text-gray-900 mb-2">Best For:</div>
                <ul className="text-gray-700 space-y-1">
                  <li>✓ Most reliable predictions</li>
                  <li>✓ Production trading systems</li>
                  <li>✓ Risk-averse strategies</li>
                  <li>✓ All market conditions</li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-r from-orange-50 to-amber-50 border-2 border-orange-300 rounded-lg p-5">
              <div className="flex items-center mb-3">
                <span className="text-3xl mr-3">🤖</span>
                <h4 className="text-xl font-bold text-orange-900">Reinforcement Learning</h4>
              </div>
              <p className="text-gray-700 mb-3">
                RL agents learn optimal trading policies through trial and error.
                Adapts to changing market conditions and optimizes for profit maximization.
              </p>
              <div className="bg-white rounded p-3 text-sm">
                <div className="font-semibold text-gray-900 mb-2">Best For:</div>
                <ul className="text-gray-700 space-y-1">
                  <li>✓ Automated trading strategies</li>
                  <li>✓ Position sizing optimization</li>
                  <li>✓ Dynamic risk management</li>
                  <li>✓ Adaptive market response</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
            <h4 className="font-semibold text-yellow-900 mb-2">🎓 Model Selection Guide</h4>
            <div className="text-yellow-800 text-sm space-y-2">
              <p><strong>Beginners:</strong> Start with LSTM models on 1h or 4h timeframes</p>
              <p><strong>Intermediate:</strong> Experiment with Transformer models for specific pairs</p>
              <p><strong>Advanced:</strong> Use Ensemble models for production trading</p>
              <p><strong>Experts:</strong> Fine-tune RL agents with custom reward functions</p>
            </div>
          </div>
        </div>
      ),
    },
    training: {
      title: 'Training AI Models',
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 p-5 rounded">
            <h3 className="text-lg font-bold text-blue-900 mb-2">🎓 Model Training Overview</h3>
            <p className="text-blue-800">
              Training custom AI models allows you to create specialized predictors tailored to specific
              trading pairs and market conditions. The platform automates the entire training pipeline.
            </p>
          </div>

          <h4 className="font-semibold text-gray-900">📋 Step-by-Step Training Guide</h4>

          <div className="space-y-4">
            <div className="bg-white border-2 border-gray-200 rounded-lg p-5">
              <div className="flex items-start">
                <div className="bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold mr-4 flex-shrink-0">1</div>
                <div className="flex-1">
                  <h5 className="font-bold text-gray-900 mb-2">Select Training Data</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• Choose your trading symbol (e.g., BTC_USDT)</li>
                    <li>• Select timeframe (1h or 4h recommended for first model)</li>
                    <li>• Specify number of candles (minimum 100, recommended 500+)</li>
                    <li>• System will fetch historical data automatically</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white border-2 border-gray-200 rounded-lg p-5">
              <div className="flex items-start">
                <div className="bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold mr-4 flex-shrink-0">2</div>
                <div className="flex-1">
                  <h5 className="font-bold text-gray-900 mb-2">Configure Model Parameters</h5>
                  <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700">
                    <div>
                      <strong>LSTM Settings:</strong>
                      <ul className="ml-4 mt-1 space-y-1">
                        <li>• Hidden units: 64-128</li>
                        <li>• Layers: 2-3</li>
                        <li>• Sequence length: 50</li>
                        <li>• Dropout: 0.2</li>
                      </ul>
                    </div>
                    <div>
                      <strong>Training Config:</strong>
                      <ul className="ml-4 mt-1 space-y-1">
                        <li>• Epochs: 50-100</li>
                        <li>• Batch size: 32</li>
                        <li>• Learning rate: 0.001</li>
                        <li>• Train/test split: 80/20</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white border-2 border-gray-200 rounded-lg p-5">
              <div className="flex items-start">
                <div className="bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold mr-4 flex-shrink-0">3</div>
                <div className="flex-1">
                  <h5 className="font-bold text-gray-900 mb-2">Start Training</h5>
                  <p className="text-sm text-gray-700 mb-2">Click "Train Model" button and monitor progress:</p>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>✓ Training loss should decrease over time</li>
                    <li>✓ Validation metrics appear after each epoch</li>
                    <li>✓ Training typically takes 2-10 minutes depending on data size</li>
                    <li>✓ Progress bar shows current epoch and estimated completion time</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white border-2 border-gray-200 rounded-lg p-5">
              <div className="flex items-start">
                <div className="bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold mr-4 flex-shrink-0">4</div>
                <div className="flex-1">
                  <h5 className="font-bold text-gray-900 mb-2">Evaluate Performance</h5>
                  <p className="text-sm text-gray-700 mb-2">Review model metrics:</p>
                  <div className="bg-gray-50 rounded p-3 text-sm">
                    <div className="grid md:grid-cols-2 gap-2">
                      <div><strong>MAE (Mean Absolute Error):</strong> Lower is better</div>
                      <div><strong>RMSE (Root Mean Squared Error):</strong> Lower is better</div>
                      <div><strong>R² Score:</strong> Closer to 1.0 is better</div>
                      <div><strong>Directional Accuracy:</strong> % of correct up/down predictions</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white border-2 border-gray-200 rounded-lg p-5">
              <div className="flex items-start">
                <div className="bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold mr-4 flex-shrink-0">5</div>
                <div className="flex-1">
                  <h5 className="font-bold text-gray-900 mb-2">Deploy or Retrain</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li><strong>Good Performance:</strong> Save and deploy model for live predictions</li>
                    <li><strong>Poor Performance:</strong> Adjust parameters and retrain</li>
                    <li><strong>Optimization:</strong> Use Auto-tune feature to find best parameters</li>
                    <li><strong>Regular Updates:</strong> Retrain weekly to adapt to market changes</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded">
            <h4 className="font-semibold text-green-900 mb-2">✅ Training Best Practices</h4>
            <ul className="space-y-1 text-green-800 text-sm">
              <li>• Use at least 500 candles for reliable training</li>
              <li>• Higher timeframes (4h, 1d) are easier to predict than 1m, 5m</li>
              <li>• Start with default parameters before optimization</li>
              <li>• Monitor for overfitting - training loss much lower than validation loss</li>
              <li>• Save model versions and track performance over time</li>
              <li>• Test models on out-of-sample data before live deployment</li>
            </ul>
          </div>

          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
            <h4 className="font-semibold text-red-900 mb-2">⚠️ Common Pitfalls</h4>
            <ul className="space-y-1 text-red-800 text-sm">
              <li>• Too little data: Models need sufficient history to learn patterns</li>
              <li>• Overfitting: Model memorizes training data but fails on new data</li>
              <li>• Wrong timeframe: 1m data is noisy, 1d data is too sparse for some strategies</li>
              <li>• Ignoring validation: Always check performance on unseen data</li>
              <li>• One-time training: Markets evolve - retrain models regularly</li>
            </ul>
          </div>
        </div>
      ),
    },
    'workflow-beginner': {
      title: 'Beginner Trading Workflow',
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-l-4 border-green-500 p-5 rounded">
            <h3 className="text-lg font-bold text-green-900 mb-2">🌱 Your First Day with PineLab</h3>
            <p className="text-green-800">
              This workflow guides absolute beginners through using the platform for the first time.
              Follow these steps to understand the basics before moving to advanced features.
            </p>
          </div>

          <div className="bg-white border-2 border-blue-200 rounded-xl p-6 shadow-lg">
            <h4 className="text-xl font-bold text-gray-900 mb-4">📚 30-Minute Beginner Tutorial</h4>

            <div className="space-y-5">
              <div className="border-l-4 border-blue-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-blue-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">1</span>
                  <h5 className="font-bold text-gray-900">Familiarize with Interface (5 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Open the platform at http://localhost:5173</li>
                  <li>• Notice the 5 main tabs: Overview, Model Lab, Deep Learning, Features, Platform</li>
                  <li>• Top header shows Symbol selector and Timeframe selector</li>
                  <li>• Start on the Overview tab (default view)</li>
                </ul>
              </div>

              <div className="border-l-4 border-purple-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-purple-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">2</span>
                  <h5 className="font-bold text-gray-900">Select a Trading Pair (2 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Click the Symbol dropdown in the header</li>
                  <li>• Type "BTC" to filter Bitcoin pairs</li>
                  <li>• Select "BTC_USDT" (most liquid pair)</li>
                  <li>• Keep timeframe at "1h" for now</li>
                </ul>
              </div>

              <div className="border-l-4 border-green-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-green-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">3</span>
                  <h5 className="font-bold text-gray-900">Observe Market Data (5 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Look at the price chart - green candles = price up, red = price down</li>
                  <li>• Green horizontal line = Support (price bounces up)</li>
                  <li>• Red horizontal line = Resistance (price bounces down)</li>
                  <li>• Volume bars show trading activity</li>
                  <li>• Watch the chart update in real-time (updates every 10 seconds)</li>
                </ul>
              </div>

              <div className="border-l-4 border-orange-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-orange-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">4</span>
                  <h5 className="font-bold text-gray-900">Check AI Signal (5 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Find the "Signal Aggregator" card at the top</li>
                  <li>• Note the signal: BUY (green), SELL (red), or HOLD (gray)</li>
                  <li>• Check confidence percentage - higher is more reliable</li>
                  <li>• Expand "View Details" to see reasoning</li>
                  <li>• Don't trade yet - just observe how signals change over time</li>
                </ul>
              </div>

              <div className="border-l-4 border-pink-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-pink-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">5</span>
                  <h5 className="font-bold text-gray-900">Review Predictions (5 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Look at the "Predictions Panel" on the right side</li>
                  <li>• See predicted price target for next period</li>
                  <li>• Note the direction (UP/DOWN) and confidence</li>
                  <li>• Compare prediction with current price on chart</li>
                  <li>• Understand this is a forecast, not a guarantee</li>
                </ul>
              </div>

              <div className="border-l-4 border-indigo-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-indigo-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">6</span>
                  <h5 className="font-bold text-gray-900">Experiment with Symbols (5 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Try different symbols: ETH_USDT, SOL_USDT, DOGE_USDT</li>
                  <li>• Notice how signals and predictions vary by asset</li>
                  <li>• Some pairs have higher volatility than others</li>
                  <li>• More popular pairs (BTC, ETH) typically have better data</li>
                </ul>
              </div>

              <div className="border-l-4 border-teal-500 pl-4">
                <div className="flex items-center mb-2">
                  <span className="bg-teal-500 text-white text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center mr-2">7</span>
                  <h5 className="font-bold text-gray-900">Try Different Timeframes (3 min)</h5>
                </div>
                <ul className="text-sm text-gray-700 space-y-1 ml-8">
                  <li>• Change timeframe to "4h" for smoother trends</li>
                  <li>• Try "15m" for faster-moving charts</li>
                  <li>• Notice: Longer timeframes = more stable signals</li>
                  <li>• Shorter timeframes = more noise and volatility</li>
                  <li>• For learning, stick with 1h or 4h</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <h4 className="font-semibold text-blue-900 mb-2">📖 What You Learned</h4>
            <ul className="space-y-1 text-blue-800 text-sm">
              <li>✓ How to navigate the platform interface</li>
              <li>✓ How to select trading pairs and timeframes</li>
              <li>✓ Reading price charts and support/resistance levels</li>
              <li>✓ Understanding AI signals and confidence scores</li>
              <li>✓ Interpreting price predictions</li>
              <li>✓ Comparing different assets and timeframes</li>
            </ul>
          </div>

          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
            <h4 className="font-semibold text-yellow-900 mb-2">🎯 Next Steps</h4>
            <ol className="space-y-2 text-yellow-800 text-sm">
              <li><strong>Day 2-3:</strong> Observe signals for 2-3 days without trading. Track accuracy.</li>
              <li><strong>Day 4-5:</strong> Learn the Model Lab - train your first LSTM model</li>
              <li><strong>Week 2:</strong> Explore Deep Learning and Feature Engineering tabs</li>
              <li><strong>Week 3:</strong> Try paper trading (simulation) before real money</li>
              <li><strong>Week 4+:</strong> Gradually increase complexity as you gain confidence</li>
            </ol>
          </div>

          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
            <h4 className="font-semibold text-red-900 mb-2">🚫 Beginner Don'ts</h4>
            <ul className="space-y-1 text-red-800 text-sm">
              <li>✗ Don't trade real money until you've practiced for weeks</li>
              <li>✗ Don't trust AI signals blindly - always verify with chart analysis</li>
              <li>✗ Don't use maximum leverage or risk more than you can afford to lose</li>
              <li>✗ Don't expect 100% accuracy - even the best models are probabilistic</li>
              <li>✗ Don't skip learning the basics to jump into advanced features</li>
            </ul>
          </div>
        </div>
      ),
    },
    faq: {
      title: 'Frequently Asked Questions',
      content: (
        <div className="space-y-4">
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ How accurate are the AI predictions?</h4>
            <p className="text-sm text-gray-700">
              Accuracy varies by market conditions, timeframe, and model type. Well-trained models typically
              achieve 55-70% directional accuracy on 4h+ timeframes. No model is 100% accurate - always use
              risk management and combine AI insights with your own analysis.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ Which timeframe should I use?</h4>
            <p className="text-sm text-gray-700">
              For beginners: 1h or 4h. Day traders: 15m or 30m. Swing traders: 4h or 1d. Longer timeframes
              produce more reliable signals but fewer trading opportunities. Shorter timeframes have more noise
              but allow for more frequent trades.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ How often should I retrain models?</h4>
            <p className="text-sm text-gray-700">
              Retrain weekly or after significant market regime changes. Models trained on bull market data may
              underperform in bear markets. Monitor model performance metrics and retrain when accuracy drops below
              acceptable thresholds.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ What's the difference between LSTM and Transformer models?</h4>
            <p className="text-sm text-gray-700">
              LSTMs excel at learning sequential patterns and long-term trends. Transformers use attention mechanisms
              to focus on the most relevant historical periods, making them better for complex patterns. Ensemble models
              combine both for maximum reliability.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ Can I use this for live trading?</h4>
            <p className="text-sm text-gray-700">
              Yes, but only after extensive backtesting and paper trading. Connect Alpaca broker credentials in
              the backend .env file to enable order execution. Start with small position sizes and gradually scale
              as you gain confidence in your strategies.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ Why are some features not loading?</h4>
            <p className="text-sm text-gray-700">
              Deep Learning features require PyTorch installation. Trading features need Alpaca API credentials.
              Email notifications require SMTP configuration. Check the backend logs and .env file to ensure all
              required dependencies and credentials are configured.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ What data sources does the platform use?</h4>
            <p className="text-sm text-gray-700">
              Primary: Crypto.com public API for 900+ cryptocurrency pairs. Optional: TradingView webhooks for
              signal integration, Alpaca for equities data and trading. All data is cached locally in SQLite for
              performance.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ How do I interpret confidence scores?</h4>
            <p className="text-sm text-gray-700">
              &gt;80%: High confidence, strong agreement across models. 60-80%: Medium confidence, use with caution.
              &lt;60%: Low confidence, wait for better setup. Confidence represents model certainty, not guaranteed
              accuracy. Always verify with technical analysis.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ What's the minimum data needed to train a model?</h4>
            <p className="text-sm text-gray-700">
              Absolute minimum: 100 candles. Recommended: 500+ candles for reliable training. More data = better
              learning of patterns. For 1h timeframe, 500 candles = ~21 days of data. Consider market conditions
              in your training window.
            </p>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h4 className="font-bold text-gray-900 mb-2">❓ Is my data secure?</h4>
            <p className="text-sm text-gray-700">
              All data is stored locally in your SQLite database. API keys and secrets are encrypted using Fernet
              encryption. Never share your .env file or database files. Use strong JWT secrets in production. For
              additional security, use PostgreSQL with SSL in production deployments.
            </p>
          </div>
        </div>
      ),
    },
  }

  const renderContent = (sectionId, subsectionId) => {
    const key = subsectionId || sectionId
    return content[key] || (
      <div className="text-gray-500">
        <p>Content for this section is coming soon.</p>
      </div>
    )
  }

  const activeMainSection = sections.find((s) =>
    s.subsections?.some((sub) => sub.id === activeSection)
  ) || sections.find((s) => s.id === activeSection)

  const filteredSections = searchTerm
    ? sections.filter((section) =>
        section.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        section.subsections?.some((sub) =>
          sub.title.toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
    : sections

  return (
    <div className="h-[calc(100vh-200px)] flex gap-6">
      {/* Sidebar Navigation */}
      <div className="w-80 bg-white border-r border-gray-200 overflow-y-auto flex-shrink-0">
        <div className="p-4 border-b border-gray-200 sticky top-0 bg-white z-10">
          <h2 className="text-xl font-bold text-gray-900 mb-3">📚 User Handbook</h2>
          <input
            type="text"
            placeholder="Search handbook..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="p-2">
          {filteredSections.map((section) => (
            <div key={section.id} className="mb-2">
              <button
                onClick={() => setActiveSection(section.id)}
                className={`w-full text-left px-3 py-2 rounded-lg font-medium text-sm transition-colors ${
                  activeMainSection?.id === section.id
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-gray-700 hover:bg-gray-50'
                }`}
              >
                <span className="mr-2">{section.icon}</span>
                {section.title}
              </button>

              {section.subsections && activeMainSection?.id === section.id && (
                <div className="ml-6 mt-1 space-y-1">
                  {section.subsections.map((sub) => (
                    <button
                      key={sub.id}
                      onClick={() => setActiveSection(sub.id)}
                      className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${
                        activeSection === sub.id
                          ? 'bg-blue-100 text-blue-800 font-medium'
                          : 'text-gray-600 hover:bg-gray-50'
                      }`}
                    >
                      {sub.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-y-auto">
        <div className="p-8">
          {content[activeSection] ? (
            <>
              <h1 className="text-3xl font-bold text-gray-900 mb-6">
                {content[activeSection].title}
              </h1>
              <div className="prose prose-blue max-w-none">
                {content[activeSection].content}
              </div>
            </>
          ) : (
            <div className="text-center py-20 text-gray-500">
              <p className="text-xl">Select a topic from the sidebar to get started</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
