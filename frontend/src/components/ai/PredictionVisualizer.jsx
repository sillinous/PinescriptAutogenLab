import React, { useState, useEffect } from 'react';
import { Card } from '../common/Card';

const PredictionVisualizer = ({ trainedModels }) => {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [predictions, setPredictions] = useState({
    lstm: null,
    transformer: null,
    ensemble: null
  });
  const [loading, setLoading] = useState(false);
  const [modelsList, setModelsList] = useState({ lstm: [], transformer: [], ensemble: [] });

  // Get unique tickers from all trained models
  const availableTickers = [
    ...new Set([
      ...trainedModels.lstm.map(m => m.ticker),
      ...trainedModels.transformer.map(m => m.ticker),
      ...trainedModels.ensemble.map(m => m.ticker)
    ])
  ];

  useEffect(() => {
    fetchModelsList();
  }, []);

  const fetchModelsList = async () => {
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/v2/deep-learning/models/list`);
      if (response.ok) {
        const data = await response.json();
        setModelsList(data);
      }
    } catch (err) {
      console.error('Failed to fetch models list:', err);
    }
  };

  const fetchPredictions = async () => {
    if (!selectedTicker) return;

    setLoading(true);
    const newPredictions = { lstm: null, transformer: null, ensemble: null };

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    // Fetch LSTM prediction
    if (trainedModels.lstm.some(m => m.ticker === selectedTicker)) {
      try {
        const response = await fetch(`${API_URL}/api/v2/deep-learning/lstm/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: selectedTicker,
            model_type: 'lstm',
            timeframe: '1h',
            sequence_length: 60
          })
        });
        if (response.ok) {
          newPredictions.lstm = await response.json();
        }
      } catch (err) {
        console.error('LSTM prediction failed:', err);
      }
    }

    // Fetch Transformer prediction
    if (trainedModels.transformer.some(m => m.ticker === selectedTicker)) {
      try {
        const response = await fetch(`${API_URL}/api/v2/deep-learning/transformer/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: selectedTicker,
            model_type: 'transformer',
            timeframe: '1h',
            sequence_length: 60
          })
        });
        if (response.ok) {
          newPredictions.transformer = await response.json();
        }
      } catch (err) {
        console.error('Transformer prediction failed:', err);
      }
    }

    // Fetch Ensemble prediction
    if (trainedModels.ensemble.some(m => m.ticker === selectedTicker)) {
      try {
        const response = await fetch(`${API_URL}/api/v2/deep-learning/ensemble/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: selectedTicker,
            model_type: 'ensemble',
            timeframe: '1h',
            sequence_length: 60
          })
        });
        if (response.ok) {
          newPredictions.ensemble = await response.json();
        }
      } catch (err) {
        console.error('Ensemble prediction failed:', err);
      }
    }

    setPredictions(newPredictions);
    setLoading(false);
  };

  const handleTickerChange = (ticker) => {
    setSelectedTicker(ticker);
    setPredictions({ lstm: null, transformer: null, ensemble: null });
  };

  const renderPredictionCard = (title, prediction, color) => {
    if (!prediction) {
      return (
        <Card>
          <h3 className={`text-lg font-semibold mb-3 text-${color}-600`}>{title}</h3>
          <p className="text-gray-400 text-sm">No prediction available</p>
        </Card>
      );
    }

    return (
      <Card>
        <h3 className={`text-lg font-semibold mb-3 text-${color}-600`}>{title}</h3>
        <div className="space-y-3">
          <div>
            <p className="text-sm text-gray-600">Predicted Price</p>
            <p className="text-2xl font-bold">
              ${prediction.prediction[0]?.toFixed(2) || 'N/A'}
            </p>
          </div>

          {prediction.confidence !== undefined && (
            <div>
              <p className="text-sm text-gray-600">Confidence</p>
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className={`bg-${color}-600 h-2 rounded-full`}
                    style={{ width: `${(prediction.confidence * 100).toFixed(0)}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}

          {prediction.uncertainty && (
            <div>
              <p className="text-sm text-gray-600 mb-1">Uncertainty Metrics</p>
              <div className="text-xs space-y-1">
                <div className="flex justify-between">
                  <span>Std Dev:</span>
                  <span className="font-mono">
                    {prediction.uncertainty.std_dev?.[0]?.toFixed(4) || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Variance:</span>
                  <span className="font-mono">
                    {prediction.uncertainty.variance?.[0]?.toFixed(4) || 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="pt-2 border-t text-xs text-gray-500">
            <p>Timestamp: {new Date(prediction.timestamp).toLocaleString()}</p>
            <p>Sequence Length: {prediction.sequence_length}</p>
            <p>Horizon: {prediction.prediction_horizon}</p>
          </div>
        </div>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {/* Ticker Selection */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Select Ticker for Predictions</h3>

        {availableTickers.length === 0 ? (
          <p className="text-gray-500">
            No trained models available. Please train models first.
          </p>
        ) : (
          <div className="flex items-center space-x-4">
            <select
              value={selectedTicker}
              onChange={(e) => handleTickerChange(e.target.value)}
              className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">-- Select Ticker --</option>
              {availableTickers.map(ticker => (
                <option key={ticker} value={ticker}>{ticker}</option>
              ))}
            </select>

            <button
              onClick={fetchPredictions}
              disabled={!selectedTicker || loading}
              className={`px-6 py-2 rounded-lg font-medium ${
                !selectedTicker || loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {loading ? 'Loading...' : '📊 Get Predictions'}
            </button>
          </div>
        )}
      </Card>

      {/* Predictions Grid */}
      {selectedTicker && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {renderPredictionCard('LSTM Prediction', predictions.lstm, 'blue')}
          {renderPredictionCard('Transformer Prediction', predictions.transformer, 'purple')}
          {renderPredictionCard('Ensemble Prediction', predictions.ensemble, 'green')}
        </div>
      )}

      {/* Comparison View */}
      {selectedTicker && predictions.lstm && predictions.transformer && (
        <Card>
          <h3 className="text-lg font-semibold mb-4">Model Comparison</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
              <span className="font-medium">LSTM</span>
              <span className="text-xl font-bold text-blue-600">
                ${predictions.lstm.prediction[0]?.toFixed(2)}
              </span>
            </div>

            <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
              <span className="font-medium">Transformer</span>
              <span className="text-xl font-bold text-purple-600">
                ${predictions.transformer.prediction[0]?.toFixed(2)}
              </span>
            </div>

            {predictions.ensemble && (
              <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                <span className="font-medium">Ensemble (Combined)</span>
                <span className="text-xl font-bold text-green-600">
                  ${predictions.ensemble.prediction[0]?.toFixed(2)}
                </span>
              </div>
            )}

            {predictions.lstm && predictions.transformer && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Prediction Difference</p>
                <p className="text-lg font-bold">
                  ${Math.abs(
                    predictions.lstm.prediction[0] - predictions.transformer.prediction[0]
                  ).toFixed(2)}
                  <span className="text-sm text-gray-500 ml-2">
                    ({(
                      (Math.abs(predictions.lstm.prediction[0] - predictions.transformer.prediction[0]) /
                        predictions.lstm.prediction[0]) *
                      100
                    ).toFixed(2)}% divergence)
                  </span>
                </p>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Models Status */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Available Models</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <h4 className="font-medium text-blue-600 mb-2">LSTM Models</h4>
            {modelsList.lstm.length === 0 ? (
              <p className="text-sm text-gray-500">No models</p>
            ) : (
              <ul className="space-y-1">
                {modelsList.lstm.map((model, idx) => (
                  <li key={idx} className="text-sm">
                    {model.ticker} ({(model.size_mb).toFixed(1)} MB)
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div>
            <h4 className="font-medium text-purple-600 mb-2">Transformer Models</h4>
            {modelsList.transformer.length === 0 ? (
              <p className="text-sm text-gray-500">No models</p>
            ) : (
              <ul className="space-y-1">
                {modelsList.transformer.map((model, idx) => (
                  <li key={idx} className="text-sm">
                    {model.ticker} ({(model.size_mb).toFixed(1)} MB)
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div>
            <h4 className="font-medium text-green-600 mb-2">Ensemble Models</h4>
            {modelsList.ensemble.length === 0 ? (
              <p className="text-sm text-gray-500">No ensembles</p>
            ) : (
              <ul className="space-y-1">
                {modelsList.ensemble.map((model, idx) => (
                  <li key={idx} className="text-sm">{model}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PredictionVisualizer;
