"""
Signal Normalization for CANShield
===================================

Purpose:
    - Normalize CAN signals to [0, 1] range using MinMax scaling
    - Compute normalization parameters from TRAINING data only
    - Apply same parameters to test/deployment data (avoid data leakage!)
    - Support inverse transformation for visualization

Critical:
    NEVER compute min/max from test data - this causes data leakage!
    Always fit() on training data, then transform() on test/deployment.

Example:
    >>> normalizer = SignalNormalizer(signal_names)
    >>> normalizer.fit(training_data)  # Compute min/max from training ONLY
    >>> normalizer.save_parameters('min_max_values.csv')
    >>> 
    >>> # Later, in deployment:
    >>> normalizer.load_parameters('min_max_values.csv')
    >>> normalized = normalizer.transform(new_data)

Reference:
    CANShield: Deep-Learning-Based Intrusion Detection Framework for 
    Controller Area Networks at the Signal Level
    IEEE Internet of Things Journal, 2023

Author: CANShield Implementation (Maritime Version)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union
import warnings


class SignalNormalizer:
    """
    MinMax scaler for CAN signals.
    
    Normalizes signals to [0, 1] range: x_norm = (x - min) / (max - min)
    
    Critical Design:
        - fit() computes min/max from TRAINING data only
        - transform() applies these parameters to any data
        - Parameters saved/loaded to ensure consistency across train/test/deploy
    
    Attributes:
        signal_names: List of signal names in order
        min_values: Dictionary mapping signal → min value
        max_values: Dictionary mapping signal → max value
        is_fitted: Whether fit() has been called
    """
    
    def __init__(self, signal_names: List[str]):
        """
        Initialize normalizer.
        
        Args:
            signal_names: List of signal names (must match data order)
        """
        self.signal_names = signal_names
        self.num_signals = len(signal_names)
        
        # Normalization parameters (set by fit())
        self.min_values = {signal: None for signal in signal_names}
        self.max_values = {signal: None for signal in signal_names}
        
        # State
        self.is_fitted = False
    
    
    def fit(self, data: np.ndarray, epsilon: float = 1e-8) -> 'SignalNormalizer':
        """
        Compute min/max from training data.
        
        Args:
            data: Training data of shape (num_signals, num_timesteps) or
                  (num_samples, num_signals, num_timesteps)
            epsilon: Small value to avoid division by zero (default: 1e-8)
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If data has wrong shape
        
        Example:
            >>> training_data = np.random.randn(15, 10000)
            >>> normalizer.fit(training_data)
            >>> print(normalizer.min_values['pitch'])  # -2.5
            >>> print(normalizer.max_values['pitch'])  # +3.8
        """
        # Handle different input shapes
        if data.ndim == 2:
            # Shape: (num_signals, num_timesteps)
            if data.shape[0] != self.num_signals:
                raise ValueError(
                    f"Expected {self.num_signals} signals, got {data.shape[0]}"
                )
            signal_data = data
            
        elif data.ndim == 3:
            # Shape: (num_samples, num_signals, num_timesteps)
            # Flatten to (num_signals, total_timesteps)
            if data.shape[1] != self.num_signals:
                raise ValueError(
                    f"Expected {self.num_signals} signals, got {data.shape[1]}"
                )
            # Reshape: (samples, signals, time) → (signals, samples*time)
            signal_data = data.transpose(1, 0, 2).reshape(self.num_signals, -1)
            
        else:
            raise ValueError(
                f"Data must be 2D or 3D, got shape {data.shape}"
            )
        
        # Compute min/max for each signal
        for i, signal in enumerate(self.signal_names):
            signal_values = signal_data[i, :]
            
            # Remove NaN values before computing min/max
            valid_values = signal_values[~np.isnan(signal_values)]
            
            if len(valid_values) == 0:
                warnings.warn(
                    f"Signal '{signal}' has no valid values, using defaults [0, 1]"
                )
                self.min_values[signal] = 0.0
                self.max_values[signal] = 1.0
            else:
                self.min_values[signal] = float(np.min(valid_values))
                self.max_values[signal] = float(np.max(valid_values))
                
                # Check for constant signals
                if abs(self.max_values[signal] - self.min_values[signal]) < epsilon:
                    warnings.warn(
                        f"Signal '{signal}' is constant (min={self.min_values[signal]:.3f}, "
                        f"max={self.max_values[signal]:.3f}), adding epsilon"
                    )
                    self.max_values[signal] = self.min_values[signal] + epsilon
        
        self.is_fitted = True
        return self
    
    
    def transform(self, data: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Normalize data to [0, 1] using pre-computed min/max.
        
        Formula: x_norm = (x - min) / (max - min)
        
        Args:
            data: Data to normalize (same shape as fit data)
            clip: Whether to clip values to [0, 1] (handles out-of-range)
        
        Returns:
            Normalized data in [0, 1] range
        
        Raises:
            RuntimeError: If fit() hasn't been called
            ValueError: If data shape doesn't match
        
        Example:
            >>> normalized = normalizer.transform(test_data)
            >>> assert normalized.min() >= 0 and normalized.max() <= 1
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Handle different input shapes
        if data.ndim == 2:
            # Shape: (num_signals, num_timesteps)
            if data.shape[0] != self.num_signals:
                raise ValueError(
                    f"Expected {self.num_signals} signals, got {data.shape[0]}"
                )
            normalized = self._normalize_2d(data, clip)
            
        elif data.ndim == 3:
            # Shape: (num_samples, num_signals, num_timesteps)
            if data.shape[1] != self.num_signals:
                raise ValueError(
                    f"Expected {self.num_signals} signals, got {data.shape[1]}"
                )
            normalized = self._normalize_3d(data, clip)
            
        else:
            raise ValueError(
                f"Data must be 2D or 3D, got shape {data.shape}"
            )
        
        return normalized
    
    
    def _normalize_2d(self, data: np.ndarray, clip: bool) -> np.ndarray:
        """
        Normalize 2D data (num_signals, num_timesteps).
        """
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for i, signal in enumerate(self.signal_names):
            min_val = self.min_values[signal]
            max_val = self.max_values[signal]
            
            # Apply normalization: (x - min) / (max - min)
            normalized[i, :] = (data[i, :] - min_val) / (max_val - min_val)
        
        # Clip to [0, 1] if requested
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    
    def _normalize_3d(self, data: np.ndarray, clip: bool) -> np.ndarray:
        """
        Normalize 3D data (num_samples, num_signals, num_timesteps).
        """
        num_samples = data.shape[0]
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for i, signal in enumerate(self.signal_names):
            min_val = self.min_values[signal]
            max_val = self.max_values[signal]
            
            # Apply to all samples: (x - min) / (max - min)
            normalized[:, i, :] = (data[:, i, :] - min_val) / (max_val - min_val)
        
        # Clip to [0, 1] if requested
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized
    
    
    def fit_transform(self, data: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Fit to data, then transform it (convenience method).
        
        Equivalent to: normalizer.fit(data).transform(data)
        
        Args:
            data: Training data
            clip: Whether to clip to [0, 1]
        
        Returns:
            Normalized data
        """
        return self.fit(data).transform(data, clip=clip)
    
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        
        Formula: x = x_norm * (max - min) + min
        
        Args:
            normalized_data: Data in [0, 1] range
        
        Returns:
            Data in original scale
        
        Raises:
            RuntimeError: If fit() hasn't been called
        
        Example:
            >>> original = normalizer.inverse_transform(normalized)
            >>> # Use for visualization or comparison
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before inverse_transform()")
        
        # Handle different shapes
        if normalized_data.ndim == 2:
            return self._inverse_transform_2d(normalized_data)
        elif normalized_data.ndim == 3:
            return self._inverse_transform_3d(normalized_data)
        else:
            raise ValueError(
                f"Data must be 2D or 3D, got shape {normalized_data.shape}"
            )
    
    
    def _inverse_transform_2d(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform 2D data."""
        original = np.zeros_like(data)
        
        for i, signal in enumerate(self.signal_names):
            min_val = self.min_values[signal]
            max_val = self.max_values[signal]
            original[i, :] = data[i, :] * (max_val - min_val) + min_val
        
        return original
    
    
    def _inverse_transform_3d(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform 3D data."""
        original = np.zeros_like(data)
        
        for i, signal in enumerate(self.signal_names):
            min_val = self.min_values[signal]
            max_val = self.max_values[signal]
            original[:, i, :] = data[:, i, :] * (max_val - min_val) + min_val
        
        return original
    
    
    def save_parameters(self, filepath: Union[str, Path]) -> None:
        """
        Save normalization parameters to CSV.
        
        CRITICAL: Save after fit() on training data, load in deployment!
        
        Args:
            filepath: Path to save CSV (e.g., 'min_max_values.csv')
        
        Example:
            >>> normalizer.fit(training_data)
            >>> normalizer.save_parameters('results/normalization/min_max.csv')
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before saving parameters")
        
        # Create DataFrame
        df = pd.DataFrame({
            'signal': self.signal_names,
            'min': [self.min_values[s] for s in self.signal_names],
            'max': [self.max_values[s] for s in self.signal_names]
        })
        
        # Save to CSV
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        
        print(f"✅ Normalization parameters saved to {filepath}")
    
    
    def load_parameters(self, filepath: Union[str, Path]) -> 'SignalNormalizer':
        """
        Load normalization parameters from CSV.
        
        CRITICAL: Use this in test/deployment to apply same normalization!
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            self (for method chaining)
        
        Example:
            >>> # In deployment
            >>> normalizer = SignalNormalizer(signal_names)
            >>> normalizer.load_parameters('results/normalization/min_max.csv')
            >>> normalized = normalizer.transform(deployment_data)
        """
        df = pd.read_csv(filepath)
        
        # Validate signals match
        loaded_signals = df['signal'].tolist()
        if loaded_signals != self.signal_names:
            warnings.warn(
                f"Loaded signals {loaded_signals} don't match initialized signals {self.signal_names}"
            )
        
        # Load min/max
        for _, row in df.iterrows():
            signal = row['signal']
            if signal in self.signal_names:
                self.min_values[signal] = float(row['min'])
                self.max_values[signal] = float(row['max'])
        
        self.is_fitted = True
        print(f"✅ Normalization parameters loaded from {filepath}")
        
        return self
    
    
    def get_parameters(self) -> pd.DataFrame:
        """
        Get normalization parameters as DataFrame.
        
        Returns:
            DataFrame with columns: signal, min, max
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() or load_parameters() first")
        
        return pd.DataFrame({
            'signal': self.signal_names,
            'min': [self.min_values[s] for s in self.signal_names],
            'max': [self.max_values[s] for s in self.signal_names]
        })
    
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"SignalNormalizer(signals={self.num_signals}, {status})"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_signal_order(filepath: Union[str, Path]) -> List[str]:
    """
    Load signal order from file.
    
    Args:
        filepath: Path to signal_order.txt
    
    Returns:
        List of signal names
    """
    with open(filepath, 'r') as f:
        signals = [line.strip() for line in f if line.strip()]
    return signals


def normalize_csv_data(csv_path: Union[str, Path],
                      signal_order_path: Union[str, Path],
                      output_path: Optional[Union[str, Path]] = None,
                      save_params: bool = True) -> tuple:
    """
    Load CSV, normalize it, and optionally save.
    
    Args:
        csv_path: Path to CSV with signal columns
        signal_order_path: Path to signal_order.txt
        output_path: Where to save normalized CSV (optional)
        save_params: Whether to save min/max parameters
    
    Returns:
        (normalized_data, normalizer, original_df)
    """
    # Load signal order
    signal_names = load_signal_order(signal_order_path)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Extract signals in correct order
    data = df[signal_names].values.T  # Shape: (num_signals, num_timesteps)
    
    # Create and fit normalizer
    normalizer = SignalNormalizer(signal_names)
    normalized = normalizer.fit_transform(data)
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        
        # Create normalized DataFrame
        normalized_df = df.copy()
        for i, signal in enumerate(signal_names):
            normalized_df[signal] = normalized[i, :]
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_df.to_csv(output_path, index=False)
        print(f"✅ Normalized data saved to {output_path}")
        
        # Save parameters
        if save_params:
            params_path = output_path.parent / 'min_max_values.csv'
            normalizer.save_parameters(params_path)
    
    return normalized, normalizer, df


# ==============================================================================
# TESTING AND DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Signal Normalization - Testing")
    print("=" * 80)
    
    # Test 1: Basic normalization
    print("\n[Test 1] Basic Normalization")
    print("-" * 80)
    
    # Create test data: 5 signals, 100 timesteps
    signal_names = ['pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle']
    np.random.seed(42)
    
    # Different ranges for each signal
    test_data = np.array([
        np.random.randn(100) * 10 + 5,    # pitch: ~[-15, 25]
        np.random.randn(100) * 20 + 100,  # depth: ~[40, 160]
        np.random.randn(100) * 5 + 15,    # wind_speed: ~[0, 30]
        np.random.randn(100) * 30 + 180,  # yaw: ~[90, 270]
        np.random.randn(100) * 45 + 180,  # wind_angle: ~[45, 315]
    ])
    
    print(f"Test data shape: {test_data.shape}")
    print("\nOriginal ranges:")
    for i, signal in enumerate(signal_names):
        print(f"  {signal:15s}: [{test_data[i].min():7.2f}, {test_data[i].max():7.2f}]")
    
    # Create normalizer
    normalizer = SignalNormalizer(signal_names)
    print(f"\nCreated: {normalizer}")
    
    # Fit and transform
    normalized = normalizer.fit_transform(test_data)
    
    print(f"\nNormalized data shape: {normalized.shape}")
    print("\nNormalized ranges:")
    for i, signal in enumerate(signal_names):
        print(f"  {signal:15s}: [{normalized[i].min():.6f}, {normalized[i].max():.6f}]")
    
    # Verify in [0, 1]
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    print("\n✅ All values in [0, 1] range!")
    
    # Test 2: Inverse transformation
    print("\n[Test 2] Inverse Transformation")
    print("-" * 80)
    
    reconstructed = normalizer.inverse_transform(normalized)
    
    print("Checking reconstruction accuracy...")
    max_error = np.abs(test_data - reconstructed).max()
    print(f"Maximum reconstruction error: {max_error:.10f}")
    assert max_error < 1e-4  # Tolerance for float32 precision
    print("✅ Inverse transform correct (within float32 precision)!")
    
    # Test 3: Save and load parameters
    print("\n[Test 3] Save and Load Parameters")
    print("-" * 80)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        param_file = f.name
    
    # Save
    normalizer.save_parameters(param_file)
    
    # Create new normalizer and load
    normalizer2 = SignalNormalizer(signal_names)
    normalizer2.load_parameters(param_file)
    
    # Verify same parameters
    print("\nComparing parameters:")
    for signal in signal_names:
        min1 = normalizer.min_values[signal]
        min2 = normalizer2.min_values[signal]
        max1 = normalizer.max_values[signal]
        max2 = normalizer2.max_values[signal]
        
        assert abs(min1 - min2) < 1e-10
        assert abs(max1 - max2) < 1e-10
    
    print("✅ Parameters saved and loaded correctly!")
    
    # Clean up
    import os
    os.remove(param_file)
    
    # Test 4: 3D data (multiple samples)
    print("\n[Test 4] 3D Data (Batch Normalization)")
    print("-" * 80)
    
    # Create batch data: 10 samples × 5 signals × 50 timesteps
    batch_data = np.random.randn(10, 5, 50) * 10 + 50
    
    print(f"Batch data shape: {batch_data.shape}")
    
    # Fit on batch
    normalizer3 = SignalNormalizer(signal_names)
    normalized_batch = normalizer3.fit_transform(batch_data)
    
    print(f"Normalized batch shape: {normalized_batch.shape}")
    print(f"Normalized range: [{normalized_batch.min():.6f}, {normalized_batch.max():.6f}]")
    
    assert normalized_batch.shape == batch_data.shape
    assert normalized_batch.min() >= 0.0 and normalized_batch.max() <= 1.0
    print("✅ Batch normalization works!")
    
    # Test 5: Get parameters as DataFrame
    print("\n[Test 5] Parameter DataFrame")
    print("-" * 80)
    
    params_df = normalizer.get_parameters()
    print("\nNormalization parameters:")
    print(params_df.to_string(index=False))
    
    # Test 6: Handle NaN values
    print("\n[Test 6] Handle NaN Values")
    print("-" * 80)
    
    # Create data with NaN
    data_with_nan = test_data.copy()
    data_with_nan[0, 10:20] = np.nan  # Add NaN to pitch signal
    
    print(f"Data with {np.isnan(data_with_nan).sum()} NaN values")
    
    normalizer_nan = SignalNormalizer(signal_names)
    normalized_nan = normalizer_nan.fit_transform(data_with_nan)
    
    print(f"After normalization: {np.isnan(normalized_nan).sum()} NaN values")
    print("✅ NaN handling works!")
    
    # Test 7: Realistic maritime signals
    print("\n[Test 7] Maritime CAN Signals")
    print("-" * 80)
    
    maritime_signals = [
        'pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle',
        'heading', 'cog', 'variation', 'sog', 'longitude',
        'rudder_angle_order', 'latitude', 'roll', 'rate_of_turn', 'rudder_position'
    ]
    
    # Realistic ranges for maritime signals
    maritime_data = np.array([
        np.random.randn(1000) * 5 + 0,      # pitch: -15° to +15°
        np.random.randn(1000) * 10 + 50,    # depth: 30m to 70m
        np.random.randn(1000) * 3 + 10,     # wind_speed: 4 to 16 m/s
        np.random.randn(1000) * 10 + 90,    # yaw: 70° to 110°
        np.random.randn(1000) * 30 + 180,   # wind_angle: 120° to 240°
        np.random.randn(1000) * 5 + 90,     # heading: 80° to 100°
        np.random.randn(1000) * 5 + 90,     # cog: 80° to 100°
        np.random.randn(1000) * 2 + 0,      # variation: -6° to +6°
        np.random.randn(1000) * 2 + 10,     # sog: 6 to 14 knots
        np.random.randn(1000) * 0.1 + 5.0,  # longitude: 4.7° to 5.3°
        np.random.randn(1000) * 5 + 0,      # rudder_angle_order: -15° to +15°
        np.random.randn(1000) * 0.1 + 45.0, # latitude: 44.7° to 45.3°
        np.random.randn(1000) * 3 + 0,      # roll: -9° to +9°
        np.random.randn(1000) * 1 + 0,      # rate_of_turn: -3 to +3 deg/s
        np.random.randn(1000) * 5 + 0,      # rudder_position: -15° to +15°
    ])
    
    print(f"Maritime data: {len(maritime_signals)} signals × 1000 timesteps")
    
    # Normalize
    maritime_normalizer = SignalNormalizer(maritime_signals)
    maritime_normalized = maritime_normalizer.fit_transform(maritime_data)
    
    print("\nOriginal ranges (selected signals):")
    for signal in ['pitch', 'depth', 'latitude', 'longitude', 'sog']:
        idx = maritime_signals.index(signal)
        print(f"  {signal:15s}: [{maritime_data[idx].min():8.3f}, {maritime_data[idx].max():8.3f}]")
    
    print("\nNormalized to [0, 1]:")
    for signal in ['pitch', 'depth', 'latitude', 'longitude', 'sog']:
        idx = maritime_signals.index(signal)
        print(f"  {signal:15s}: [{maritime_normalized[idx].min():.6f}, {maritime_normalized[idx].max():.6f}]")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Normalization module working correctly.")
    print("=" * 80)
    
    # Summary
    print("\n[SUMMARY] Signal Normalization")
    print("-" * 80)
    print("✅ MinMax scaling to [0, 1] range")
    print("✅ Fit on training data, transform test/deployment")
    print("✅ Save/load parameters to avoid data leakage")
    print("✅ Inverse transformation for visualization")
    print("✅ Handle 2D and 3D data (batches)")
    print("✅ NaN-aware (ignores NaN in min/max computation)")
    print("✅ Ready for CNN autoencoder training!")
    print("=" * 80)
