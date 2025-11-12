"""
Data Loader and Preprocessor for CANShield
===========================================

Purpose:
    - Load and preprocess CAN data for CNN training
    - Apply full preprocessing pipeline:
        1. Load CSV → Extract signals
        2. Forward-fill missing values
        3. Generate multi-scale views
        4. Normalize to [0, 1]
        5. Reshape for CNN (add channel dimension)
    - Generate training/test datasets

Output Format:
    - Shape: (num_samples, num_signals, window_size, 1)
    - Example: (1000, 15, 50, 1) for 1000 windows of 15 signals × 50 timesteps

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
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Import our preprocessing modules
try:
    from .multi_scale import MultiScaleGenerator
    from .normalization import SignalNormalizer
    from .forward_fill import ForwardFillProcessor
except ImportError:
    # For standalone testing
    import sys
    sys.path.append(str(Path(__file__).parent))
    from multi_scale import MultiScaleGenerator
    from normalization import SignalNormalizer
    from forward_fill import ForwardFillProcessor


class CANDataLoader:
    """
    Complete data loader for CANShield preprocessing pipeline.
    
    Applies all preprocessing steps:
        CSV → Forward-fill → Multi-scale → Normalize → Reshape
    
    Attributes:
        signal_names: List of signals in optimal order
        sampling_periods: List of sampling periods [1, 5, 10, 20, 50]
        window_size: Size of each window (default: 50)
        normalizer: SignalNormalizer instance (one per sampling period)
    """
    
    def __init__(self,
                 signal_names: List[str],
                 sampling_periods: List[int] = [1, 5, 10, 20, 50],
                 window_size: int = 50):
        """
        Initialize data loader.
        
        Args:
            signal_names: List of signal names (from signal_order.txt)
            sampling_periods: Sampling periods for multi-scale views
            window_size: Size of each window
        """
        self.signal_names = signal_names
        self.num_signals = len(signal_names)
        self.sampling_periods = sampling_periods
        self.window_size = window_size
        
        # Create multi-scale generator
        self.multi_scale = MultiScaleGenerator(sampling_periods, window_size)
        
        # Create normalizers (one per sampling period)
        self.normalizers = {
            T: SignalNormalizer(signal_names) for T in sampling_periods
        }
        
        # State
        self.is_fitted = False
    
    
    def load_and_preprocess(self,
                           csv_path: Union[str, Path],
                           apply_bfill: bool = True,
                           stride: int = 1) -> Dict[int, np.ndarray]:
        """
        Load CSV and apply full preprocessing pipeline.
        
        Args:
            csv_path: Path to CSV with decoded CAN signals
            apply_bfill: Whether to apply backward-fill (training only!)
            stride: Stride for sliding windows (1 = overlapping)
        
        Returns:
            Dictionary mapping sampling_period → windows array
            Each array has shape: (num_windows, num_signals, window_size, 1)
        
        Steps:
            1. Load CSV
            2. Extract signals in correct order
            3. Forward-fill missing values
            4. Backward-fill if training (can see future in CSV!)
            5. Generate multi-scale views with sliding windows
            6. Return windows (normalization done separately via fit/transform)
        
        Example:
            >>> loader = CANDataLoader(signal_names)
            >>> windows = loader.load_and_preprocess('training_data.csv')
            >>> windows[1].shape   # (num_windows, 15, 50, 1) for T=1
            >>> windows[5].shape   # (num_windows, 15, 50, 1) for T=5
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Extract signals in correct order
        signal_data = df[self.signal_names].copy()
        
        # Apply forward-fill
        signal_data = signal_data.ffill()
        
        # Apply backward-fill if training (ONLY for offline training!)
        if apply_bfill:
            signal_data = signal_data.bfill()
        
        # Drop any remaining NaN
        signal_data = signal_data.dropna()
        
        # Convert to numpy: (num_timesteps, num_signals) → (num_signals, num_timesteps)
        data = signal_data.values.T
        
        print(f"Loaded data shape: {data.shape}")
        print(f"  Signals: {data.shape[0]}")
        print(f"  Timesteps: {data.shape[1]}")
        
        # Generate multi-scale sliding windows
        all_windows = self.multi_scale.generate_sliding_windows(data, stride=stride)
        
        # Reshape each view: (num_windows, num_signals, window_size) → (num_windows, num_signals, window_size, 1)
        reshaped_windows = {}
        for T, windows in all_windows.items():
            # Add channel dimension
            reshaped = windows[..., np.newaxis]  # Add last dimension
            reshaped_windows[T] = reshaped
            print(f"  T={T:2d}: {reshaped.shape[0]} windows of shape {reshaped.shape[1:]}")
        
        return reshaped_windows
    
    
    def fit_normalizers(self, windows_dict: Dict[int, np.ndarray]) -> 'CANDataLoader':
        """
        Fit normalizers on training data.
        
        CRITICAL: Only call this on TRAINING data!
        
        Args:
            windows_dict: Dictionary from load_and_preprocess()
        
        Returns:
            self (for method chaining)
        
        Example:
            >>> train_windows = loader.load_and_preprocess('train.csv')
            >>> loader.fit_normalizers(train_windows)
            >>> loader.save_normalizers('results/normalization/')
        """
        for T in self.sampling_periods:
            if T in windows_dict:
                # Remove channel dimension for fitting: (N, signals, time, 1) → (N, signals, time)
                data = windows_dict[T].squeeze(-1)
                self.normalizers[T].fit(data)
                print(f"✅ Fitted normalizer for T={T}")
        
        self.is_fitted = True
        return self
    
    
    def transform_windows(self, windows_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Normalize windows using fitted parameters.
        
        Args:
            windows_dict: Dictionary from load_and_preprocess()
        
        Returns:
            Dictionary with normalized windows
        
        Example:
            >>> test_windows = loader.load_and_preprocess('test.csv')
            >>> normalized = loader.transform_windows(test_windows)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit_normalizers() before transform_windows()")
        
        normalized_dict = {}
        for T in self.sampling_periods:
            if T in windows_dict:
                # Remove channel dimension: (N, signals, time, 1) → (N, signals, time)
                data = windows_dict[T].squeeze(-1)
                
                # Normalize
                normalized = self.normalizers[T].transform(data)
                
                # Add channel dimension back: (N, signals, time) → (N, signals, time, 1)
                normalized = normalized[..., np.newaxis]
                
                normalized_dict[T] = normalized
                print(f"✅ Normalized T={T}: {normalized.shape}")
        
        return normalized_dict
    
    
    def fit_transform(self, windows_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Fit and transform in one step (training data only).
        
        Args:
            windows_dict: Dictionary from load_and_preprocess()
        
        Returns:
            Normalized windows
        """
        return self.fit_normalizers(windows_dict).transform_windows(windows_dict)
    
    
    def save_normalizers(self, output_dir: Union[str, Path]) -> None:
        """
        Save normalization parameters for all sampling periods.
        
        Args:
            output_dir: Directory to save CSV files
        
        Example:
            >>> loader.save_normalizers('results/normalization/')
            # Saves: min_max_T1.csv, min_max_T5.csv, ...
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit_normalizers() before saving")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for T in self.sampling_periods:
            filepath = output_dir / f'min_max_T{T}.csv'
            self.normalizers[T].save_parameters(filepath)
    
    
    def load_normalizers(self, input_dir: Union[str, Path]) -> 'CANDataLoader':
        """
        Load normalization parameters from saved files.
        
        Args:
            input_dir: Directory with saved CSV files
        
        Returns:
            self (for method chaining)
        
        Example:
            >>> # In deployment
            >>> loader = CANDataLoader(signal_names)
            >>> loader.load_normalizers('results/normalization/')
            >>> test_windows = loader.load_and_preprocess('test.csv', apply_bfill=False)
            >>> normalized = loader.transform_windows(test_windows)
        """
        input_dir = Path(input_dir)
        
        for T in self.sampling_periods:
            filepath = input_dir / f'min_max_T{T}.csv'
            if filepath.exists():
                self.normalizers[T].load_parameters(filepath)
            else:
                warnings.warn(f"Normalizer file not found: {filepath}")
        
        self.is_fitted = True
        return self
    
    
    def save_windows(self,
                     windows_dict: Dict[int, np.ndarray],
                     output_dir: Union[str, Path],
                     prefix: str = 'train') -> None:
        """
        Save processed windows to .npy files.
        
        Args:
            windows_dict: Normalized windows dictionary
            output_dir: Directory to save files
            prefix: Filename prefix ('train', 'test', etc.)
        
        Example:
            >>> loader.save_windows(train_windows, 'data/processed/', 'train')
            # Saves: train_T1.npy, train_T5.npy, ...
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for T, windows in windows_dict.items():
            filepath = output_dir / f'{prefix}_T{T}.npy'
            np.save(filepath, windows)
            print(f"✅ Saved {filepath}: {windows.shape}")
    
    
    def load_windows(self,
                     input_dir: Union[str, Path],
                     prefix: str = 'train') -> Dict[int, np.ndarray]:
        """
        Load processed windows from .npy files.
        
        Args:
            input_dir: Directory with saved files
            prefix: Filename prefix
        
        Returns:
            Dictionary of windows
        """
        input_dir = Path(input_dir)
        windows_dict = {}
        
        for T in self.sampling_periods:
            filepath = input_dir / f'{prefix}_T{T}.npy'
            if filepath.exists():
                windows = np.load(filepath)
                windows_dict[T] = windows
                print(f"✅ Loaded {filepath}: {windows.shape}")
            else:
                warnings.warn(f"File not found: {filepath}")
        
        return windows_dict
    
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"CANDataLoader(\n"
            f"  signals={self.num_signals},\n"
            f"  sampling_periods={self.sampling_periods},\n"
            f"  window_size={self.window_size},\n"
            f"  {status}\n"
            f")"
        )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_signal_order(filepath: Union[str, Path]) -> List[str]:
    """Load signal order from file, skipping comments and empty lines."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.strip().startswith('#')]


def prepare_training_data(csv_path: Union[str, Path],
                          signal_order_path: Union[str, Path],
                          output_dir: Union[str, Path],
                          sampling_periods: List[int] = [1, 5, 10, 20, 50],
                          window_size: int = 50,
                          stride: int = 10) -> CANDataLoader:
    """
    Complete pipeline: Load → Preprocess → Normalize → Save.
    
    Use this for training data preparation.
    
    Args:
        csv_path: Path to training CSV
        signal_order_path: Path to signal_order.txt
        output_dir: Where to save processed data
        sampling_periods: Sampling periods
        window_size: Window size
        stride: Stride for windows
    
    Returns:
        Fitted CANDataLoader instance
    
    Example:
        >>> loader = prepare_training_data(
        ...     'data/training.csv',
        ...     'results/initialization/signal_order.txt',
        ...     'data/processed/'
        ... )
    """
    print("=" * 80)
    print("TRAINING DATA PREPARATION")
    print("=" * 80)
    
    # Load signal order
    signal_names = load_signal_order(signal_order_path)
    print(f"\nLoaded {len(signal_names)} signals from {signal_order_path}")
    
    # Create loader
    loader = CANDataLoader(signal_names, sampling_periods, window_size)
    print(f"\n{loader}")
    
    # Load and preprocess
    print(f"\nLoading data from {csv_path}...")
    windows = loader.load_and_preprocess(csv_path, apply_bfill=True, stride=stride)
    
    # Fit normalizers
    print("\nFitting normalizers on training data...")
    loader.fit_normalizers(windows)
    
    # Transform
    print("\nNormalizing windows...")
    normalized = loader.transform_windows(windows)
    
    # Save everything
    output_dir = Path(output_dir)
    print(f"\nSaving to {output_dir}...")
    loader.save_normalizers(output_dir / 'normalization')
    loader.save_windows(normalized, output_dir / 'windows', 'train')
    
    print("\n✅ Training data preparation complete!")
    print("=" * 80)
    
    return loader


# ==============================================================================
# TESTING AND DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Loader - Testing")
    print("=" * 80)
    
    # Test 1: Create synthetic CSV
    print("\n[Test 1] Create Synthetic Training Data")
    print("-" * 80)
    
    signal_names = ['pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle']
    num_timesteps = 2000
    
    # Generate realistic maritime data
    np.random.seed(42)
    synthetic_data = {
        'pitch': np.random.randn(num_timesteps) * 5 + 0,
        'depth': np.random.randn(num_timesteps) * 10 + 50,
        'wind_speed': np.random.randn(num_timesteps) * 3 + 10,
        'yaw': np.random.randn(num_timesteps) * 10 + 90,
        'wind_angle': np.random.randn(num_timesteps) * 30 + 180,
    }
    
    df = pd.DataFrame(synthetic_data)
    
    # Add some NaN to test forward/backward fill
    df.loc[10:15, 'pitch'] = np.nan
    df.loc[50:55, 'wind_speed'] = np.nan
    
    print(f"Created synthetic data: {df.shape}")
    print(f"NaN count: {df.isna().sum().sum()}")
    
    # Save to temp CSV
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_file = f.name
    
    df.to_csv(csv_file, index=False)
    print(f"Saved to: {csv_file}")
    
    # Test 2: Load and preprocess
    print("\n[Test 2] Load and Preprocess")
    print("-" * 80)
    
    loader = CANDataLoader(signal_names, sampling_periods=[1, 5, 10], window_size=50)
    print(f"\n{loader}")
    
    windows = loader.load_and_preprocess(csv_file, apply_bfill=True, stride=20)
    
    print(f"\nGenerated windows:")
    for T, w in windows.items():
        print(f"  T={T}: {w.shape}")
    
    # Test 3: Fit normalizers
    print("\n[Test 3] Fit Normalizers")
    print("-" * 80)
    
    loader.fit_normalizers(windows)
    
    # Test 4: Transform
    print("\n[Test 4] Normalize Windows")
    print("-" * 80)
    
    normalized = loader.transform_windows(windows)
    
    print("\nNormalized windows:")
    for T, w in normalized.items():
        print(f"  T={T}: shape={w.shape}, range=[{w.min():.6f}, {w.max():.6f}]")
    
    # Verify in [0, 1]
    for T, w in normalized.items():
        assert w.min() >= 0.0 and w.max() <= 1.0
    print("\n✅ All windows normalized to [0, 1]!")
    
    # Test 5: Save and load normalizers
    print("\n[Test 5] Save and Load Normalizers")
    print("-" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        loader.save_normalizers(tmpdir)
        
        # Create new loader and load
        loader2 = CANDataLoader(signal_names, sampling_periods=[1, 5, 10], window_size=50)
        loader2.load_normalizers(tmpdir)
        
        # Transform with loaded normalizers
        normalized2 = loader2.transform_windows(windows)
        
        # Verify same results
        for T in [1, 5, 10]:
            diff = np.abs(normalized[T] - normalized2[T]).max()
            assert diff < 1e-6
        
        print("✅ Save/load normalizers works correctly!")
    
    # Test 6: Save and load windows
    print("\n[Test 6] Save and Load Windows")
    print("-" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        loader.save_windows(normalized, tmpdir, 'test')
        
        # Load
        loaded = loader.load_windows(tmpdir, 'test')
        
        # Verify
        for T in [1, 5, 10]:
            assert np.allclose(normalized[T], loaded[T])
        
        print("✅ Save/load windows works correctly!")
    
    # Test 7: Verify CNN input shape
    print("\n[Test 7] Verify CNN Input Shape")
    print("-" * 80)
    
    for T, windows in normalized.items():
        print(f"\nT={T} windows:")
        print(f"  Shape: {windows.shape}")
        print(f"  Expected: (num_samples, {len(signal_names)}, 50, 1)")
        
        assert windows.ndim == 4
        assert windows.shape[1] == len(signal_names)
        assert windows.shape[2] == 50
        assert windows.shape[3] == 1
    
    print("\n✅ All windows have correct CNN input shape!")
    
    # Clean up
    import os
    os.remove(csv_file)
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Data loader working correctly.")
    print("=" * 80)
    
    # Summary
    print("\n[SUMMARY] Data Loader")
    print("-" * 80)
    print("✅ Complete preprocessing pipeline implemented")
    print("✅ Load CSV → Forward-fill → Backward-fill (training)")
    print("✅ Generate multi-scale sliding windows")
    print("✅ Normalize to [0, 1] with separate parameters per view")
    print("✅ Reshape for CNN: (N, signals, time, 1)")
    print("✅ Save/load normalizers (avoid data leakage)")
    print("✅ Save/load processed windows (.npy files)")
    print("✅ Ready for CNN autoencoder training!")
    print("=" * 80)
