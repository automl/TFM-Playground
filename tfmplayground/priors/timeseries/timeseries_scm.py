"""Time Series Structural Causal Model for synthetic data generation.

This module provides TimeSeriesSCM, which generates (X, y) pairs where
the target y[t] depends only on past features X[t'] for t' < t.

Unlike TabICL's MLPSCM which treats rows as exchangeable, this SCM
respects temporal causality for forecasting tasks.
"""

from __future__ import annotations

import random
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn, Tensor

from .temporal_sampler import TemporalXSampler


class TimeSeriesSCM(nn.Module):
    """Structural Causal Model for time series data generation.
    
    Generates synthetic tabular datasets where:
    - Features X have temporal correlations (via TemporalXSampler)
    - Target y[t] depends only on X[t'] for t' < t (temporal causality)
    - Optionally creates lag features
    
    Parameters
    ----------
    seq_len : int, default=1024
        Number of time steps (rows) to generate.
    
    num_features : int, default=10
        Number of base features.
    
    num_outputs : int, default=1
        Number of output dimensions (typically 1 for univariate forecasting).
    
    include_lags : bool, default=True
        Whether to include lag features in X.
    
    max_lags : int, default=5
        Maximum number of lag features per base feature.
    
    num_layers : int, default=1
        Number of MLP layers for the transformation. Default is 1 (simple).
    
    hidden_dim : int, default=32
        Hidden dimension of MLP layers (only used if num_layers > 1).
    
    activation : str, default="tanh"
        Activation function: "tanh", "relu", or "identity".
    
    noise_std : float, default=0.1
        Standard deviation of noise added to outputs.
    
    device : str, default="cpu"
        Device to place tensors on.
    
    Examples
    --------
    >>> scm = TimeSeriesSCM(seq_len=100, num_features=5)
    >>> X, y = scm()
    >>> X.shape, y.shape
    (torch.Size([100, 5]), torch.Size([100]))
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 10,
        num_outputs: int = 1,
        include_lags: bool = True,
        max_lags: int = 5,
        num_layers: int = 1,  # Default to simple (Option C)
        hidden_dim: int = 32,
        activation: str = "tanh",
        noise_std: float = 0.1,
        device: str = "cpu",
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.include_lags = include_lags
        self.max_lags = max_lags
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.noise_std = noise_std
        self.device = device
        
        # Calculate input dimension for the MLP
        if include_lags:
            self.input_dim = num_features * (max_lags + 1)
        else:
            self.input_dim = num_features
        
        # Initialize temporal sampler (pass kwargs for temporal pattern config)
        self.sampler = TemporalXSampler(
            seq_len=seq_len,
            num_features=num_features,
            device=device,
            **{k: v for k, v in kwargs.items() if k in ['trend_prob', 'seasonal_prob', 'ar_prob']},
        )
        
        # Build MLP layers
        self._build_layers()
    
    def _build_layers(self):
        """Build the MLP transformation layers."""
        # Get activation function
        if self.activation_name == "tanh":
            activation = nn.Tanh()
        elif self.activation_name == "relu":
            activation = nn.ReLU()
        else:  # identity
            activation = nn.Identity()
        
        if self.num_layers == 1:
            # Simple linear transformation
            self.layers = nn.Linear(self.input_dim, self.num_outputs)
        else:
            # Multi-layer MLP
            layers = []
            in_dim = self.input_dim
            
            for i in range(self.num_layers - 1):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(activation)
                in_dim = self.hidden_dim
            
            layers.append(nn.Linear(in_dim, self.num_outputs))
            self.layers = nn.Sequential(*layers)
        
        # Initialize weights randomly
        self._init_weights()
        
        # Move to device
        self.layers = self.layers.to(self.device)
    
    def _init_weights(self):
        """Initialize weights with random values."""
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                # Random initialization with reasonable scale
                nn.init.normal_(module.weight, mean=0, std=1.0 / (module.in_features ** 0.5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self) -> Tuple[Tensor, Tensor]:
        """Generate a synthetic time series dataset.
        
        Returns
        -------
        X : Tensor
            Features of shape (seq_len, total_features) where total_features
            is num_features * (1 + max_lags) if include_lags=True.
        
        y : Tensor
            Target of shape (seq_len,).
        """
        # Generate base features with temporal correlations
        X_base = self.sampler.sample()  # (seq_len, num_features)
        
        # Create lag features if requested
        if self.include_lags:
            X = self.create_lag_features(X_base)
        else:
            X = X_base
        
        # Generate target from features
        y = self.generate_target(X)
        
        # Check for NaNs and handle them
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.zeros(self.seq_len, device=self.device)
        
        return X, y
    
    def create_lag_features(self, X: Tensor) -> Tensor:
        """Create lagged versions of features.
        
        Parameters
        ----------
        X : Tensor
            Original features of shape (seq_len, num_features).
        
        Returns
        -------
        X_with_lags : Tensor
            Features with lags of shape (seq_len, num_features * (1 + max_lags)).
            Lagged values before the start are filled with the first value.
        """
        seq_len, num_features = X.shape
        all_features = [X]  # Original features (lag 0)
        
        for lag in range(1, self.max_lags + 1):
            # Shift features by lag positions
            lagged = torch.zeros_like(X)
            lagged[lag:] = X[:-lag]
            # Fill initial values with the first observation (avoid NaN)
            lagged[:lag] = X[0:1].expand(lag, -1)
            all_features.append(lagged)
        
        X_with_lags = torch.cat(all_features, dim=-1)
        return X_with_lags
    
    def generate_target(self, X: Tensor) -> Tensor:
        """Generate target y from features X respecting temporal causality.
        
        y[t] is computed from X[t-1] (one step lag) to ensure no data leakage.
        
        Parameters
        ----------
        X : Tensor
            Features of shape (seq_len, total_features).
        
        Returns
        -------
        y : Tensor
            Target of shape (seq_len,).
        """
        seq_len = X.shape[0]
        y = torch.zeros(seq_len, device=self.device)
        
        with torch.no_grad():
            for t in range(1, seq_len):
                # Use features from previous time step (temporal causality)
                x_input = X[t - 1:t]  # (1, features)
                y_pred = self.layers(x_input)  # (1, num_outputs)
                y[t] = y_pred.squeeze()
            
            # First value: use some default or sample from noise
            y[0] = y[1] if seq_len > 1 else 0.0
        
        # Add noise
        y = y + torch.randn(seq_len, device=self.device) * self.noise_std
        
        # Normalize target
        y = (y - y.mean()) / (y.std() + 1e-8)
        
        return y
