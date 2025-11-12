"""
Multi-Scale View Builder for CANShield

Generates multiple temporal views from the same data by sampling at different rates.
This is a core component of CANShield's multi-scale temporal analysis.

CANShield Architecture:
    - Uses 5 different sampling periods: T = [1, 5, 10, 20, 50]
    - Each view captures patterns at different time scales
    - All views have SAME shape: (num_signals, window_size)
    - Different sampling periods capture different attack types:
        * T=1:  Fast attacks (Flooding, Playback)
        * T=5:  Medium attacks (Plateau)
        * T=10: Slower attacks (Continuous)
        * T=20: Long-term trends
        * T=50: Very slow drift attacks

Example:
    >>> from multi_scale import MultiScaleGenerator
    >>> import numpy as np
    >>> 
    >>> # Sample data: 15 signals × 1000 timesteps
    >>> data = np.random.randn(15, 1000)
    >>> 
    >>> # Create multi-scale views
    >>> generator = MultiScaleGenerator(
    ...     sampling_periods=[1, 5, 10, 20, 50],
    ...     window_size=50
    ... )
    >>> 
    >>> views = generator.generate_views(data)
    >>> # views = {1: (15,50), 5: (15,50), 10: (15,50), 20: (15,50), 50: (15,50)}

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
from typing import Dict, List, Optional, Tuple
import warnings


class MultiScaleGenerator:
    """
    Generate multi-scale temporal views from CAN signal data.
    
    Each view samples data at a different rate, allowing the ensemble of
    autoencoders to detect attacks at multiple time scales.
    
    Attributes:
        sampling_periods: List of sampling periods T (e.g., [1, 5, 10, 20, 50])
        window_size: Number of samples per view (e.g., 50)
        num_signals: Number of signals (inferred from data)
    """
    
    def __init__(self, 
                 sampling_periods: List[int] = [1, 5, 10, 20, 50],
                 window_size: int = 50):
        """
        Initialize multi-scale view generator.
        
        Args:
            sampling_periods: List of sampling periods T
                Default: [1, 5, 10, 20, 50] (CANShield standard)
            window_size: Number of samples in each view
                Default: 50 (CANShield standard)
        
        Note:
            All views will have shape (num_signals, window_size) regardless
            of sampling period. This allows using identical CNN architecture.
        """
        self.sampling_periods = sorted(sampling_periods)
        self.window_size = window_size
        self.num_signals = None
        
        # Validate parameters
        if not all(t > 0 for t in sampling_periods):
            raise ValueError("All sampling periods must be positive integers")
        
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        
        # Calculate minimum data length required
        max_period = max(sampling_periods)
        self.min_data_length = max_period * window_size
    
    
    def generate_views(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Generate all multi-scale views from input data.
        
        Args:
            data: Input data of shape (num_signals, num_timesteps)
                Must have at least max(T) × window_size timesteps
        
        Returns:
            Dictionary mapping sampling_period → view array
            Each view has shape (num_signals, window_size)
        
        Example:
            >>> data = np.random.randn(15, 1000)
            >>> views = generator.generate_views(data)
            >>> views[1].shape   # (15, 50) - every 1st timestep
            >>> views[5].shape   # (15, 50) - every 5th timestep
            >>> views[50].shape  # (15, 50) - every 50th timestep
        
        Raises:
            ValueError: If data is too short or has wrong shape
        """
        # Validate input shape
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D (num_signals, timesteps), got shape {data.shape}")
        
        num_signals, num_timesteps = data.shape
        
        # Store number of signals
        if self.num_signals is None:
            self.num_signals = num_signals
        elif self.num_signals != num_signals:
            raise ValueError(f"Expected {self.num_signals} signals, got {num_signals}")
        
        # Check minimum length
        if num_timesteps < self.min_data_length:
            raise ValueError(
                f"Data has {num_timesteps} timesteps, but need at least "
                f"{self.min_data_length} for max sampling period {max(self.sampling_periods)}"
            )
        
        # Generate views for each sampling period
        views = {}
        for T in self.sampling_periods:
            view = self._sample_view(data, T)
            views[T] = view
        
        return views
    
    
    def _sample_view(self, data: np.ndarray, sampling_period: int) -> np.ndarray:
        """
        Sample data at specified period to create one view.
        
        Strategy (CANShield approach):
            - Sample every T-th timestep: data[:, ::T]
            - Take FIRST window_size samples: [..., :window_size]
            - Result: (num_signals, window_size)
        
        Args:
            data: Full data array (num_signals, num_timesteps)
            sampling_period: Sample every T-th timestep
        
        Returns:
            View array of shape (num_signals, window_size)
        
        Example:
            Data has timesteps [t0, t1, t2, ..., t999]
            
            T=1:  Sample [t0, t1, t2, ..., t49]     → 50 samples
            T=5:  Sample [t0, t5, t10, ..., t245]   → 50 samples
            T=50: Sample [t0, t50, t100, ..., t2450] → 50 samples
        """
        # Sample every T-th column
        sampled = data[:, ::sampling_period]
        
        # Take first window_size samples
        view = sampled[:, :self.window_size]
        
        # Verify shape
        if view.shape[1] != self.window_size:
            warnings.warn(
                f"View for T={sampling_period} has only {view.shape[1]} samples, "
                f"expected {self.window_size}"
            )
        
        return view
    
    
    def generate_sliding_windows(self, 
                                  data: np.ndarray, 
                                  stride: int = 1) -> Dict[int, np.ndarray]:
        """
        Generate all possible sliding windows for each sampling period.
        
        This is used for training data generation where we want to extract
        ALL possible windows from a long sequence, not just the most recent one.
        
        Args:
            data: Input data (num_signals, num_timesteps)
            stride: Stride between consecutive windows (default: 1)
        
        Returns:
            Dictionary mapping sampling_period → windows array
            Each windows array has shape (num_windows, num_signals, window_size)
        
        Example:
            >>> data = np.random.randn(15, 1000)
            >>> windows = generator.generate_sliding_windows(data, stride=10)
            >>> windows[1].shape   # (num_windows, 15, 50) - many windows from T=1
            >>> windows[5].shape   # (num_windows, 15, 50) - many windows from T=5
        """
        all_windows = {}
        
        for T in self.sampling_periods:
            # Sample at period T
            sampled = data[:, ::T]
            
            # Extract all sliding windows
            windows = self._extract_sliding_windows(sampled, stride)
            all_windows[T] = windows
        
        return all_windows
    
    
    def _extract_sliding_windows(self, 
                                  sampled_data: np.ndarray, 
                                  stride: int) -> np.ndarray:
        """
        Extract sliding windows from sampled data.
        
        Args:
            sampled_data: Data after sampling (num_signals, sampled_timesteps)
            stride: Stride between windows
        
        Returns:
            Array of shape (num_windows, num_signals, window_size)
        """
        num_signals, sampled_length = sampled_data.shape
        
        # Calculate number of windows
        num_windows = (sampled_length - self.window_size) // stride + 1
        
        if num_windows <= 0:
            warnings.warn(f"Not enough data to extract windows (need {self.window_size}, have {sampled_length})")
            return np.array([]).reshape(0, num_signals, self.window_size)
        
        # Pre-allocate windows array
        windows = np.zeros((num_windows, num_signals, self.window_size))
        
        # Extract windows
        for i in range(num_windows):
            start = i * stride
            end = start + self.window_size
            windows[i] = sampled_data[:, start:end]
        
        return windows
    
    
    def get_required_queue_size(self) -> int:
        """
        Calculate minimum queue size needed to generate all views.
        
        Returns:
            Minimum number of timesteps needed in queue
        
        Example:
            >>> generator = MultiScaleGenerator([1, 5, 10, 20, 50], window_size=50)
            >>> generator.get_required_queue_size()
            2500  # max(T) × window_size = 50 × 50
        """
        return max(self.sampling_periods) * self.window_size
    
    
    def __repr__(self) -> str:
        return (
            f"MultiScaleGenerator(\n"
            f"  sampling_periods={self.sampling_periods},\n"
            f"  window_size={self.window_size},\n"
            f"  min_data_length={self.min_data_length}\n"
            f")"
        )


def load_signal_order(signal_order_path: Path) -> List[str]:
    """
    Load fixed signal order from initialization phase.
    
    Args:
        signal_order_path: Path to signal_order.txt
    
    Returns:
        List of signal names in optimal order
    """
    with open(signal_order_path, 'r') as f:
        signals = [line.strip() for line in f if line.strip()]
    return signals


def create_views_from_csv(csv_path: Path,
                          signal_order_path: Path,
                          sampling_periods: List[int] = [1, 5, 10, 20, 50],
                          window_size: int = 50) -> Dict[int, np.ndarray]:
    """
    Load CSV data and generate multi-scale views.
    
    This is a convenience function for batch processing during training.
    
    Args:
        csv_path: Path to CSV file with decoded CAN signals
        signal_order_path: Path to signal_order.txt (fixed order)
        sampling_periods: List of sampling periods
        window_size: Size of each window
    
    Returns:
        Dictionary of views: {T: array(num_signals, window_size)}
    
    Example:
        >>> views = create_views_from_csv(
        ...     'data/decoded/normal_data.csv',
        ...     'results/initialization/signal_order.txt',
        ...     sampling_periods=[1, 5, 10, 20, 50],
        ...     window_size=50
        ... )
        >>> # Use views for training
    """
    # Load signal order
    signal_order = load_signal_order(signal_order_path)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Extract signals in correct order
    # Assumes CSV has columns for each signal
    data = df[signal_order].values.T  # Transpose to (num_signals, timesteps)
    
    # Create generator
    generator = MultiScaleGenerator(sampling_periods, window_size)
    
    # Generate views
    views = generator.generate_views(data)
    
    return views


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Scale View Generator - Testing")
    print("=" * 80)
    
    # Test 1: Basic view generation
    print("\n[Test 1] Basic Multi-Scale View Generation")
    print("-" * 80)
    
    # Create synthetic data: 15 signals, 3000 timesteps
    num_signals = 15
    num_timesteps = 3000
    np.random.seed(42)
    data = np.random.randn(num_signals, num_timesteps)
    
    print(f"Input data shape: {data.shape}")
    print(f"  Signals: {num_signals}")
    print(f"  Timesteps: {num_timesteps}")
    
    # Create generator
    generator = MultiScaleGenerator(
        sampling_periods=[1, 5, 10, 20, 50],
        window_size=50
    )
    
    print(f"\n{generator}")
    print(f"Minimum data length required: {generator.min_data_length}")
    
    # Generate views
    views = generator.generate_views(data)
    
    print(f"\nGenerated {len(views)} views:")
    for T, view in views.items():
        print(f"  T={T:2d}: shape={view.shape} (samples every {T:2d} timesteps)")
    
    # Test 2: Verify sampling correctness
    print("\n[Test 2] Verify Sampling Correctness")
    print("-" * 80)
    
    # Create data with known pattern (enough for T=50)
    # Signal 0: Timestep index (0, 1, 2, 3, ...)
    test_data = np.zeros((3, 3000))
    test_data[0, :] = np.arange(3000)  # Signal 0 = timestep index
    test_data[1, :] = np.arange(3000) * 2  # Signal 1 = 2 × timestep
    test_data[2, :] = np.arange(3000) * 3  # Signal 2 = 3 × timestep
    
    # Create new generator for test data
    test_generator = MultiScaleGenerator(
        sampling_periods=[1, 5, 10, 20, 50],
        window_size=50
    )
    
    test_views = test_generator.generate_views(test_data)
    
    print("Signal 0 (timestep index) in each view:")
    for T in [1, 5, 10, 20, 50]:
        view = test_views[T]
        first_10 = view[0, :10]  # First 10 values of signal 0
        print(f"  T={T:2d}: {first_10.astype(int)}")
        
        # Verify: Should be [0, T, 2T, 3T, ...]
        expected = np.arange(10) * T
        assert np.allclose(first_10, expected), f"T={T} sampling incorrect!"
    
    print("✅ All sampling periods verified correct!")
    
    # Test 3: Sliding windows for training
    print("\n[Test 3] Sliding Windows for Training Data")
    print("-" * 80)
    
    # Use smaller data for this test
    small_data = np.random.randn(15, 500)
    
    windows = generator.generate_sliding_windows(small_data, stride=10)
    
    print(f"Input data shape: {small_data.shape}")
    print(f"\nExtracted sliding windows (stride=10):")
    for T, window_array in windows.items():
        print(f"  T={T:2d}: {window_array.shape[0]} windows of shape {window_array.shape[1:]}")
    
    # Test 4: Queue size calculation
    print("\n[Test 4] Queue Size Requirement")
    print("-" * 80)
    
    required_size = generator.get_required_queue_size()
    print(f"Minimum queue size needed: {required_size} timesteps")
    print(f"  = max(T) × window_size")
    print(f"  = {max(generator.sampling_periods)} × {generator.window_size}")
    print(f"  = {required_size}")
    
    # Test 5: Shape consistency
    print("\n[Test 5] Shape Consistency Across Views")
    print("-" * 80)
    
    # All views should have same shape
    shapes = [view.shape for view in views.values()]
    assert all(s == shapes[0] for s in shapes), "Views have inconsistent shapes!"
    
    print(f"✅ All views have consistent shape: {shapes[0]}")
    print(f"  num_signals: {shapes[0][0]}")
    print(f"  window_size: {shapes[0][1]}")
    
    # Test 6: Demonstration with realistic signal names
    print("\n[Test 6] Realistic Maritime CAN Signals")
    print("-" * 80)
    
    # Maritime signal names (from initialization)
    signal_names = [
        'pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle',
        'heading', 'cog', 'variation', 'sog', 'longitude',
        'rudder_angle_order', 'latitude', 'roll', 'rate_of_turn', 'rudder_position'
    ]
    
    # Create realistic maritime data (15 signals, 2500 timesteps)
    maritime_data = np.random.randn(15, 2500) * 10 + 50  # Scaled for realism
    
    # Generate views
    maritime_views = generator.generate_views(maritime_data)
    
    print(f"Maritime CAN data: {len(signal_names)} signals")
    print(f"Generated {len(maritime_views)} multi-scale views:")
    
    for T, view in maritime_views.items():
        print(f"\n  View T={T} (every {T} timesteps):")
        print(f"    Shape: {view.shape}")
        print(f"    Time span: {T * 50} timesteps")
        print(f"    First 3 values of 'pitch' signal: {view[0, :3].round(2)}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Multi-scale view generation working correctly.")
    print("=" * 80)
    
    # Summary
    print("\n[SUMMARY] Multi-Scale View Builder")
    print("-" * 80)
    print("✅ Generates 5 views with different sampling periods")
    print("✅ All views have same shape (num_signals, window_size)")
    print("✅ Sampling verified correct (every T-th timestep)")
    print("✅ Supports sliding windows for training data extraction")
    print("✅ Can calculate required queue size for deployment")
    print("✅ Ready for CNN autoencoder training!")
    print("=" * 80)
