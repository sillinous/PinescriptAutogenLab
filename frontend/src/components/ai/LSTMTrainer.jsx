import React, { useState } from 'react';
import { Card } from '../common/Card';
import { LoadingCard } from '../common/Loading';

const LSTMTrainer = ({ onModelTrained, trainedModels }) => {
  const [config, setConfig] = useState({
    ticker: 'BTC_USDT',
    timeframe: '1h',
    lookback_days: 60,
    sequence_length: 60,
    prediction_horizon: 1,
    hidden_size: 128,
    num_layers: 2,
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    early_stopping_patience: 10
  });

  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: name.includes('learning_rate') ? parseFloat(value) : parseInt(value) || value
    }));
  };

  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    setTrainingResult(null);

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/v2/deep-learning/lstm/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Training failed');
      }

      const result = await response.json();
      setTrainingResult(result);
      onModelTrained(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  };

  const handlePredict = async (ticker) => {
    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/v2/deep-learning/lstm/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          model_type: 'lstm',
          timeframe: '1h',
          sequence_length: 60
        })
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const result = await response.json();
      alert(`LSTM Prediction for ${ticker}:\n${JSON.stringify(result.prediction, null, 2)}`);
    } catch (err) {
      alert(`Prediction error: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration Form */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">LSTM Configuration</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Basic Config */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Ticker Symbol
            </label>
            <input
              type="text"
              name="ticker"
              value={config.ticker}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Timeframe
            </label>
            <select
              name="timeframe"
              value={config.timeframe}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="1m">1 Minute</option>
              <option value="5m">5 Minutes</option>
              <option value="15m">15 Minutes</option>
              <option value="1h">1 Hour</option>
              <option value="4h">4 Hours</option>
              <option value="1d">1 Day</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Lookback Days
            </label>
            <input
              type="number"
              name="lookback_days"
              value={config.lookback_days}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Model Architecture */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Hidden Size
            </label>
            <input
              type="number"
              name="hidden_size"
              value={config.hidden_size}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Layers
            </label>
            <input
              type="number"
              name="num_layers"
              value={config.num_layers}
              onChange={handleInputChange}
              min="1"
              max="5"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Sequence Length
            </label>
            <input
              type="number"
              name="sequence_length"
              value={config.sequence_length}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Training Config */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Epochs
            </label>
            <input
              type="number"
              name="epochs"
              value={config.epochs}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Batch Size
            </label>
            <input
              type="number"
              name="batch_size"
              value={config.batch_size}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Learning Rate
            </label>
            <input
              type="number"
              name="learning_rate"
              value={config.learning_rate}
              onChange={handleInputChange}
              step="0.0001"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Train Button */}
        <div className="mt-6">
          <button
            onClick={handleTrain}
            disabled={training}
            className={`px-6 py-3 rounded-lg font-medium ${
              training
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {training ? 'Training...' : '🚀 Train LSTM Model'}
          </button>
        </div>
      </Card>

      {/* Training Status */}
      {training && (
        <LoadingCard message="Training LSTM model... This may take several minutes." />
      )}

      {/* Training Result */}
      {trainingResult && (
        <Card>
          <h3 className="text-lg font-semibold mb-4 text-green-600">
            ✅ Training Completed
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Final Train Loss</p>
              <p className="text-xl font-bold">
                {trainingResult.training_result.final_train_loss.toFixed(6)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Final Val Loss</p>
              <p className="text-xl font-bold">
                {trainingResult.training_result.final_val_loss.toFixed(6)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Best Val Loss</p>
              <p className="text-xl font-bold text-green-600">
                {trainingResult.training_result.best_val_loss.toFixed(6)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Epochs Trained</p>
              <p className="text-xl font-bold">
                {trainingResult.training_result.epochs_trained}
              </p>
            </div>
          </div>
          <p className="mt-4 text-sm text-gray-600">
            Model saved to: <code className="bg-gray-100 px-2 py-1 rounded">
              {trainingResult.model_path}
            </code>
          </p>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <Card>
          <div className="text-red-600">
            <h3 className="font-semibold mb-2">❌ Training Error</h3>
            <p>{error}</p>
          </div>
        </Card>
      )}

      {/* Trained Models List */}
      {trainedModels && trainedModels.length > 0 && (
        <Card>
          <h3 className="text-lg font-semibold mb-4">Trained Models</h3>
          <div className="space-y-2">
            {trainedModels.map((model, idx) => (
              <div
                key={idx}
                className="flex justify-between items-center p-3 bg-gray-50 rounded-lg"
              >
                <div>
                  <p className="font-medium">{model.ticker}</p>
                  <p className="text-sm text-gray-600">
                    Trained: {new Date(model.timestamp).toLocaleString()}
                  </p>
                </div>
                <button
                  onClick={() => handlePredict(model.ticker)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Predict
                </button>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default LSTMTrainer;
