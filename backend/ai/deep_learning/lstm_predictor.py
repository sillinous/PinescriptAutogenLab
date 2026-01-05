"""
LSTM-based price prediction model.

This module implements a Long Short-Term Memory (LSTM) neural network
for time series price prediction in financial markets.

Features:
- Multi-step ahead price forecasting
- Attention mechanism for improved predictions
- Bidirectional LSTM for capturing past and future context
- Dropout and regularization to prevent overfitting
- Support for multi-feature inputs (OHLCV + technical indicators)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        train: bool = True
    ):
        """
        Args:
            data: Input data of shape (timesteps, features)
            sequence_length: Number of timesteps to look back
            prediction_horizon: Number of steps to predict ahead
            train: Whether this is training data (for data splits)
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train = train

        # Create sequences
        self.sequences = []
        self.targets = []

        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length:i + sequence_length + prediction_horizon, 0]  # Predict close price
            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for price prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        prediction_horizon: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            prediction_horizon: Number of steps to predict
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Linear(
            hidden_size * self.num_directions,
            1
        )

        # Output layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, prediction_horizon)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, prediction_horizon)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)

        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights shape: (batch_size, sequence_length, 1)

        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context shape: (batch_size, hidden_size * num_directions)

        # Output layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMPredictor:
    """
    LSTM-based price predictor with training and inference capabilities.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Args:
            input_size: Number of input features (default: OHLCV = 5)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            sequence_length: Lookback window
            prediction_horizon: Steps ahead to predict
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            learning_rate: Learning rate for Adam optimizer
            device: Device to run on ('cpu' or 'cuda')
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = AttentionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_horizon=prediction_horizon,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

        # Data scaling parameters (for normalization)
        self.mean = None
        self.std = None

    def prepare_data(
        self,
        data: pd.DataFrame,
        train_split: float = 0.8,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.

        Args:
            data: DataFrame with OHLCV data
            train_split: Fraction of data to use for training
            batch_size: Batch size for training

        Returns:
            train_loader, val_loader
        """
        # Convert to numpy and normalize
        data_array = data.values.astype(np.float32)

        # Store normalization parameters
        self.mean = data_array.mean(axis=0)
        self.std = data_array.std(axis=0)

        # Normalize
        data_normalized = (data_array - self.mean) / (self.std + 1e-8)

        # Split data
        split_idx = int(len(data_normalized) * train_split)
        train_data = data_normalized[:split_idx]
        val_data = data_normalized[split_idx:]

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            train=True
        )

        val_dataset = TimeSeriesDataset(
            val_data,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            train=False
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Validate we have enough data
        if len(train_dataset) == 0:
            raise ValueError(
                f"Insufficient training data. Need at least {self.sequence_length + self.prediction_horizon} "
                f"rows but got {len(train_data)} after splitting."
            )

        if len(val_dataset) == 0:
            raise ValueError(
                f"Insufficient validation data. Need at least {self.sequence_length + self.prediction_horizon} "
                f"rows but got {len(val_data)} after splitting."
            )

        return train_loader, val_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Epochs to wait for improvement
            verbose: Print training progress

        Returns:
            Dictionary with training metrics
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)  # Prevent division by zero
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    predictions = self.model(sequences)
                    loss = self.criterion(predictions, targets)
                    val_loss += loss.item()

            val_loss /= max(len(val_loader), 1)  # Prevent division by zero
            self.val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(self.best_state)

        return {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses)
        }

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            data: Input data of shape (sequence_length, features)

        Returns:
            Predictions of shape (prediction_horizon,)
        """
        self.model.eval()

        # Normalize
        if self.mean is not None and self.std is not None:
            data_normalized = (data - self.mean) / (self.std + 1e-8)
        else:
            data_normalized = data

        # Convert to tensor
        x = torch.FloatTensor(data_normalized).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(x)

        # Denormalize
        predictions_denorm = predictions.cpu().numpy() * self.std[0] + self.mean[0]

        return predictions_denorm.flatten()

    def save(self, filepath: str):
        """Save model and configuration."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'learning_rate': self.learning_rate
            },
            'normalization': {
                'mean': self.mean.tolist() if self.mean is not None else None,
                'std': self.std.tolist() if self.std is not None else None
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
        }, filepath)

    def load(self, filepath: str):
        """Load model and configuration."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Load config
        config = checkpoint['config']
        self.__init__(**config, device=str(self.device))

        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load normalization
        if checkpoint['normalization']['mean'] is not None:
            self.mean = np.array(checkpoint['normalization']['mean'])
            self.std = np.array(checkpoint['normalization']['std'])

        # Load history
        self.train_losses = checkpoint['training_history']['train_losses']
        self.val_losses = checkpoint['training_history']['val_losses']

        self.model.eval()
