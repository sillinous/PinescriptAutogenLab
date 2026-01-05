import React, { useState } from 'react';
import { Card } from '../common/Card';
import { Section } from '../common/Section';
import LSTMTrainer from './LSTMTrainer';
import TransformerTrainer from './TransformerTrainer';
import EnsembleManager from './EnsembleManager';
import PredictionVisualizer from './PredictionVisualizer';

const DeepLearningDashboard = () => {
  const [activeTab, setActiveTab] = useState('lstm');
  const [trainedModels, setTrainedModels] = useState({
    lstm: [],
    transformer: [],
    ensemble: []
  });

  const tabs = [
    { id: 'lstm', label: 'LSTM Models', icon: '🧠' },
    { id: 'transformer', label: 'Transformers', icon: '🔮' },
    { id: 'ensemble', label: 'Ensembles', icon: '🎯' },
    { id: 'predictions', label: 'Predictions', icon: '📊' }
  ];

  const handleModelTrained = (modelType, modelData) => {
    setTrainedModels(prev => ({
      ...prev,
      [modelType]: [...prev[modelType], modelData]
    }));
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Phase 2: Deep Learning
        </h1>
        <p className="text-gray-600">
          Advanced neural networks for price prediction and pattern recognition
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-2 mb-6 border-b border-gray-200">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-3 font-medium transition-colors ${
              activeTab === tab.id
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="space-y-6">
        {activeTab === 'lstm' && (
          <Section title="LSTM Price Prediction">
            <LSTMTrainer
              onModelTrained={(data) => handleModelTrained('lstm', data)}
              trainedModels={trainedModels.lstm}
            />
          </Section>
        )}

        {activeTab === 'transformer' && (
          <Section title="Transformer-Based Forecasting">
            <TransformerTrainer
              onModelTrained={(data) => handleModelTrained('transformer', data)}
              trainedModels={trainedModels.transformer}
            />
          </Section>
        )}

        {activeTab === 'ensemble' && (
          <Section title="Ensemble Model Management">
            <EnsembleManager
              lstmModels={trainedModels.lstm}
              transformerModels={trainedModels.transformer}
              onEnsembleCreated={(data) => handleModelTrained('ensemble', data)}
            />
          </Section>
        )}

        {activeTab === 'predictions' && (
          <Section title="Prediction Visualization">
            <PredictionVisualizer trainedModels={trainedModels} />
          </Section>
        )}
      </div>

      {/* Model Status Summary */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <h3 className="font-semibold text-gray-700 mb-2">LSTM Models</h3>
          <p className="text-3xl font-bold text-blue-600">
            {trainedModels.lstm.length}
          </p>
          <p className="text-sm text-gray-500 mt-1">Trained models</p>
        </Card>

        <Card>
          <h3 className="font-semibold text-gray-700 mb-2">Transformers</h3>
          <p className="text-3xl font-bold text-purple-600">
            {trainedModels.transformer.length}
          </p>
          <p className="text-sm text-gray-500 mt-1">Trained models</p>
        </Card>

        <Card>
          <h3 className="font-semibold text-gray-700 mb-2">Ensembles</h3>
          <p className="text-3xl font-bold text-green-600">
            {trainedModels.ensemble.length}
          </p>
          <p className="text-sm text-gray-500 mt-1">Active ensembles</p>
        </Card>
      </div>
    </div>
  );
};

export default DeepLearningDashboard;
