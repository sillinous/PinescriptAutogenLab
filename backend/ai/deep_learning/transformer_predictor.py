"""
Transformer-based price prediction model.

This module implements a Transformer neural network for time series
forecasting, leveraging self-attention mechanisms to capture long-range
dependencies in price data.

Features:
- Multi-head self-attention for capturing complex patterns
- Positional encoding for temporal information
- Layer normalization and residual connections
- Support for multi-step ahead predictions
- Efficient training with teacher forcing
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional
from pathlib import Path


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for time series prediction."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        prediction_horizon: int = 1
    ):
        """
        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            prediction_horizon: Number of steps to predict
        """
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output projection
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, prediction_horizon)

    def forward(self, src):
        """
        Forward pass.

        Args:
            src: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, prediction_horizon)
        """
        # Project input to d_model dimensions
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Transformer encoding
        output = self.transformer_encoder(src)

        # Global average pooling over sequence dimension
        output = output.mean(dim=1)

        # Output projection
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output


class TransformerPredictor:
    """
    Transformer-based price predictor with training and inference.
    """

    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            sequence_length: Lookback window
            prediction_horizon: Steps ahead to predict
            dropout: Dropout probability
            learning_rate: Learning rate
            device: Device to run on
        """
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = TransformerModel(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            prediction_horizon=prediction_horizon
        ).to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []

        # Normalization parameters
        self.mean = None
        self.std = None

    def prepare_data(self, data: pd.DataFrame, train_split: float = 0.8, batch_size: int = 32):
        """Prepare data for training (reuses LSTM's TimeSeriesDataset)."""
        from .lstm_predictor import TimeSeriesDataset
        from torch.utils.data import DataLoader

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

        return train_loader, val_loader

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """Train the model."""
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
        """Make predictions on new data."""
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
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'dropout': self.dropout,
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
