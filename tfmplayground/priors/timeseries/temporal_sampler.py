"""Temporal feature sampler for time series prior generation.

This module provides TemporalXSampler, which generates feature matrices
with temporal correlations (unlike TabICL's XSampler which generates i.i.d. features).

Supported temporal patterns:
- Trends: linear, quadratic, exponential
- Seasonality: sinusoidal with configurable period
- Autoregressive: AR(p) processes
- Random walks: integrated noise
"""

from __future__ import annotations

import math
import random
from typing import Optional, List, Dict, Any

import torch
from torch import Tensor


class TemporalXSampler:
    """Generates time-correlated features for synthetic time series datasets.
    
    Unlike TabICL's XSampler which generates i.i.d. samples, this sampler
    creates features with temporal structure: trends, seasonality, and
    autoregressive patterns.
    
    Parameters
    ----------
    seq_len : int
        Number of time steps to generate.
    
    num_features : int
        Number of feature columns to generate.
    
    device : str, default="cpu"
        Device to place tensors on.
    
    trend_prob : float, default=0.3
        Probability of including a trend component.
    
    seasonal_prob : float, default=0.3
        Probability of including a seasonal component.
    
    ar_prob : float, default=0.5
        Probability of using AR process (vs random walk).
    
    Examples
    --------
    >>> sampler = TemporalXSampler(seq_len=100, num_features=5)
    >>> X = sampler.sample()
    >>> X.shape
    torch.Size([100, 5])
    """
    
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        device: str = "cpu",
        trend_prob: float = 0.3,
        seasonal_prob: float = 0.3,
        ar_prob: float = 0.5,
    ):
        self.seq_len = seq_len
        self.num_features = num_features
        self.device = device
        self.trend_prob = trend_prob
        self.seasonal_prob = seasonal_prob
        self.ar_prob = ar_prob
    
    def sample(self) -> Tensor:
        """Generate a feature matrix with temporal correlations.
        
        Each feature is generated as a mixture of temporal patterns.
        
        Returns
        -------
        X : Tensor
            Feature matrix of shape (seq_len, num_features).
        """
        features = []
        for _ in range(self.num_features):
            x = self.sample_mixed()
            features.append(x)
        
        X = torch.stack(features, dim=-1)  # (seq_len, num_features)
        return X
    
    def sample_trend(self, trend_type: str = "linear", strength: float = 1.0) -> Tensor:
        """Generate a trend component.
        
        Parameters
        ----------
        trend_type : str
            Type of trend: "linear", "quadratic", or "exponential".
        
        strength : float
            Scale of the trend.
        
        Returns
        -------
        trend : Tensor
            Trend component of shape (seq_len,).
        """
        t = torch.linspace(0, 1, self.seq_len, device=self.device)
        
        if trend_type == "linear":
            trend = t * strength
        elif trend_type == "quadratic":
            trend = (t ** 2) * strength
        elif trend_type == "exponential":
            trend = (torch.exp(t) - 1) / (math.e - 1) * strength
        else:
            trend = t * strength  # default to linear
        
        # Randomly make trend positive or negative
        if random.random() < 0.5:
            trend = -trend
        
        return trend
    
    def sample_seasonal(self, period: Optional[int] = None, amplitude: float = 1.0) -> Tensor:
        """Generate a seasonal component.
        
        Parameters
        ----------
        period : int, optional
            Period of the seasonal pattern. If None, sampled randomly.
        
        amplitude : float
            Amplitude of the seasonal oscillation.
        
        Returns
        -------
        seasonal : Tensor
            Seasonal component of shape (seq_len,).
        """
        if period is None:
            # Sample from common periods
            period = random.choice([7, 12, 24, 52, self.seq_len // 4])
        
        t = torch.arange(self.seq_len, dtype=torch.float, device=self.device)
        phase = random.uniform(0, 2 * math.pi)
        
        seasonal = amplitude * torch.sin(2 * math.pi * t / period + phase)
        
        return seasonal
    
    def sample_ar(self, order: int = 1, coefficients: Optional[List[float]] = None) -> Tensor:
        """Generate an autoregressive AR(p) process.
        
        Parameters
        ----------
        order : int
            Order of the AR process (number of lags).
        
        coefficients : list of float, optional
            AR coefficients. If None, sampled randomly ensuring stationarity.
        
        Returns
        -------
        ar : Tensor
            AR process of shape (seq_len,).
        """
        if coefficients is None:
            # Sample coefficients that sum to less than 1 (for stationarity)
            coefficients = []
            remaining = random.uniform(0.3, 0.95)
            for i in range(order):
                if i < order - 1:
                    coef = random.uniform(0, remaining * 0.8)
                else:
                    coef = remaining * random.uniform(0.5, 1.0)
                coefficients.append(coef)
                remaining -= coef
        
        # Initialize with noise
        ar = torch.zeros(self.seq_len, device=self.device)
        noise_std = random.uniform(0.1, 0.5)
        noise = torch.randn(self.seq_len, device=self.device) * noise_std
        
        # Generate AR process
        for t in range(self.seq_len):
            ar[t] = noise[t]
            for i, coef in enumerate(coefficients):
                if t - i - 1 >= 0:
                    ar[t] += coef * ar[t - i - 1]
        
        return ar
    
    def sample_random_walk(self, drift: float = 0.0, volatility: float = 1.0) -> Tensor:
        """Generate a random walk process.
        
        Parameters
        ----------
        drift : float
            Drift term (mean of increments).
        
        volatility : float
            Volatility (std of increments).
        
        Returns
        -------
        rw : Tensor
            Random walk of shape (seq_len,).
        """
        increments = torch.randn(self.seq_len, device=self.device) * volatility + drift
        rw = torch.cumsum(increments, dim=0)
        
        return rw
    
    def sample_mixed(self) -> Tensor:
        """Generate a single feature as a mixture of temporal patterns.
        
        Randomly combines trend, seasonality, AR/random walk, and noise.
        
        Returns
        -------
        x : Tensor
            Mixed temporal feature of shape (seq_len,).
        """
        x = torch.zeros(self.seq_len, device=self.device)
        
        # Base: AR process or random walk
        if random.random() < self.ar_prob:
            ar_order = random.choice([1, 2, 3])
            x = self.sample_ar(order=ar_order)
        else:
            volatility = random.uniform(0.05, 0.3)
            x = self.sample_random_walk(volatility=volatility)
        
        # Optionally add trend
        if random.random() < self.trend_prob:
            trend_type = random.choice(["linear", "quadratic"])
            trend_strength = random.uniform(0.5, 2.0)
            x = x + self.sample_trend(trend_type=trend_type, strength=trend_strength)
        
        # Optionally add seasonality
        if random.random() < self.seasonal_prob:
            amplitude = random.uniform(0.3, 1.5)
            x = x + self.sample_seasonal(amplitude=amplitude)
        
        # Add observation noise
        noise_std = random.uniform(0.01, 0.2)
        x = x + torch.randn(self.seq_len, device=self.device) * noise_std
        
        # Normalize to zero mean, unit variance
        x = (x - x.mean()) / (x.std() + 1e-8)
        
        return x
