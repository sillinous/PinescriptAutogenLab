import React, { useState } from 'react';
import { Card } from '../common/Card';
import { LoadingCard } from '../common/Loading';

const EnsembleManager = ({ lstmModels, transformerModels, onEnsembleCreated }) => {
  const [config, setConfig] = useState({
    ticker: '',
    model_types: [],
    weights: null,
    ensemble_method: 'weighted_average'
  });

  const [creating, setCreating] = useState(false);
  const [createdEnsemble, setCreatedEnsemble] = useState(null);
  const [error, setError] = useState(null);

  // Get unique tickers from trained models
  const availableTickers = [
    ...new Set([
      ...lstmModels.map(m => m.ticker),
      ...transformerModels.map(m => m.ticker)
    ])
  ];

  const handleModelTypeToggle = (modelType) => {
    setConfig(prev => ({
      ...prev,
      model_types: prev.model_types.includes(modelType)
        ? prev.model_types.filter(t => t !== modelType)
        : [...prev.model_types, modelType]
    }));
  };

  const handleCreateEnsemble = async () => {
    if (!config.ticker) {
      setError('Please select a ticker');
      return;
    }

    if (config.model_types.length === 0) {
      setError('Please select at least one model type');
      return;
    }

    setCreating(true);
    setError(null);
    setCreatedEnsemble(null);

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/v2/deep-learning/ensemble/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ensemble creation failed');
      }

      const result = await response.json();
      setCreatedEnsemble(result);
      onEnsembleCreated(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handlePredictEnsemble = async () => {
    if (!config.ticker) {
      alert('Please select a ticker first');
      return;
    }

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/v2/deep-learning/ensemble/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: config.ticker,
          model_type: 'ensemble',
          timeframe: '1h',
          sequence_length: 60
        })
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const result = await response.json();
      const predictionText = `
Ensemble Prediction for ${config.ticker}:
Prediction: ${JSON.stringify(result.prediction, null, 2)}
Confidence: ${(result.confidence * 100).toFixed(2)}%
Uncertainty: ${JSON.stringify(result.uncertainty, null, 2)}
      `.trim();
      alert(predictionText);
    } catch (err) {
      alert(`Prediction error: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Ensemble Configuration</h3>

        <div className="space-y-4">
          {/* Ticker Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Ticker
            </label>
            {availableTickers.length === 0 ? (
              <p className="text-gray-500 text-sm">
                No trained models available. Please train LSTM or Transformer models first.
              </p>
            ) : (
              <select
                value={config.ticker}
                onChange={(e) => setConfig(prev => ({ ...prev, ticker: e.target.value }))}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
              >
                <option value="">-- Select Ticker --</option>
                {availableTickers.map(ticker => (
                  <option key={ticker} value={ticker}>{ticker}</option>
                ))}
              </select>
            )}
          </div>

          {/* Model Type Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model Types to Combine
            </label>
            <div className="space-y-2">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={config.model_types.includes('lstm')}
                  onChange={() => handleModelTypeToggle('lstm')}
                  disabled={!config.ticker || !lstmModels.some(m => m.ticker === config.ticker)}
                  className="rounded text-blue-600 focus:ring-2 focus:ring-blue-500"
                />
                <span className={!config.ticker || !lstmModels.some(m => m.ticker === config.ticker) ? 'text-gray-400' : ''}>
                  LSTM Model
                  {config.ticker && lstmModels.some(m => m.ticker === config.ticker) && ' ✓'}
                </span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={config.model_types.includes('transformer')}
                  onChange={() => handleModelTypeToggle('transformer')}
                  disabled={!config.ticker || !transformerModels.some(m => m.ticker === config.ticker)}
                  className="rounded text-purple-600 focus:ring-2 focus:ring-purple-500"
                />
                <span className={!config.ticker || !transformerModels.some(m => m.ticker === config.ticker) ? 'text-gray-400' : ''}>
                  Transformer Model
                  {config.ticker && transformerModels.some(m => m.ticker === config.ticker) && ' ✓'}
                </span>
              </label>
            </div>
          </div>

          {/* Ensemble Method */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Ensemble Method
            </label>
            <select
              value={config.ensemble_method}
              onChange={(e) => setConfig(prev => ({ ...prev, ensemble_method: e.target.value }))}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
            >
              <option value="weighted_average">Weighted Average</option>
              <option value="median">Median</option>
              <option value="best">Best Model</option>
            </select>
          </div>

          {/* Create Button */}
          <button
            onClick={handleCreateEnsemble}
            disabled={creating || !config.ticker || config.model_types.length === 0}
            className={`w-full px-6 py-3 rounded-lg font-medium ${
              creating || !config.ticker || config.model_types.length === 0
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {creating ? 'Creating...' : '🎯 Create Ensemble'}
          </button>
        </div>
      </Card>

      {/* Creating Status */}
      {creating && (
        <LoadingCard message="Creating ensemble model..." />
      )}

      {/* Creation Result */}
      {createdEnsemble && (
        <Card>
          <h3 className="text-lg font-semibold mb-4 text-green-600">
            ✅ Ensemble Created Successfully
          </h3>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Ticker</p>
                <p className="font-medium">{createdEnsemble.ticker}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Number of Models</p>
                <p className="font-medium">{createdEnsemble.num_models}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Method</p>
                <p className="font-medium capitalize">{createdEnsemble.ensemble_method.replace('_', ' ')}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Created</p>
                <p className="font-medium text-sm">
                  {new Date(createdEnsemble.timestamp).toLocaleString()}
                </p>
              </div>
            </div>

            <div>
              <p className="text-sm text-gray-600 mb-1">Model Weights</p>
              <div className="flex space-x-2">
                {createdEnsemble.weights.map((weight, idx) => (
                  <div key={idx} className="px-3 py-1 bg-gray-100 rounded">
                    <span className="text-sm">
                      {createdEnsemble.model_types[idx]}: {(weight * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={handlePredictEnsemble}
              className="mt-4 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              Make Prediction
            </button>
          </div>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <Card>
          <div className="text-red-600">
            <h3 className="font-semibold mb-2">❌ Error</h3>
            <p>{error}</p>
          </div>
        </Card>
      )}

      {/* Info Card */}
      <Card>
        <h3 className="text-lg font-semibold mb-3">📚 About Ensemble Models</h3>
        <div className="space-y-2 text-sm text-gray-600">
          <p>
            Ensemble models combine predictions from multiple models to achieve more robust
            and accurate forecasts.
          </p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><strong>Weighted Average:</strong> Combines models using learned weights based on performance</li>
            <li><strong>Median:</strong> Uses the median prediction, reducing sensitivity to outliers</li>
            <li><strong>Best Model:</strong> Selects the best performing model dynamically</li>
          </ul>
          <p className="mt-3 text-xs text-gray-500">
            Train multiple models on the same ticker to create powerful ensembles.
          </p>
        </div>
      </Card>
    </div>
  );
};

export default EnsembleManager;
