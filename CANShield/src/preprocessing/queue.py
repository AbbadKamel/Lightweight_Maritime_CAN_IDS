"""
FIFO Queue for CAN Signal Management
=====================================

Purpose:
    - Buffer incoming CAN signals (asynchronous arrival)
    - Maintain sliding window of recent history (1000 timesteps)
    - Support forward-fill for missing values
    - Extract windows for CNN processing

Author: CANShield Project
Date: 2025-11-07
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import pandas as pd


class FIFOQueue:
    """
    First-In-First-Out Queue for CAN signal buffering.
    
    Manages asynchronous signal arrival and provides sliding window extraction.
    
    Attributes:
        capacity (int): Maximum number of timesteps to store
        signal_names (List[str]): Ordered list of signal names
        num_signals (int): Number of signals being tracked
        queue (deque): Circular buffer storing timesteps
        last_known_values (Dict[str, float]): Last valid value per signal (for forward-fill)
    """
    
    def __init__(self, signal_names: List[str], capacity: int = 1000):
        """
        Initialize FIFO Queue.
        
        Args:
            signal_names: List of signal names in OPTIMAL ORDER (from signal_order.txt)
            capacity: Maximum number of timesteps to store (default: 1000)
        """
        self.capacity = capacity
        self.signal_names = signal_names
        self.num_signals = len(signal_names)
        
        # Circular buffer (deque) for efficient append/pop
        self.queue = deque(maxlen=capacity)
        
        # Track last known value for each signal (for forward-fill)
        self.last_known_values = {signal: None for signal in signal_names}
        
        # Statistics
        self.total_enqueued = 0
        self.total_dequeued = 0
    
    
    def enqueue(self, timestamp: float, signal_values: Dict[str, float]) -> None:
        """
        Add new timestep to queue with available signal values.
        
        Args:
            timestamp: Timestamp of the measurement
            signal_values: Dictionary mapping signal_name -> value
                          (only for signals that arrived at this timestep)
        
        Example:
            queue.enqueue(t=1.0, {'wind_speed': 5.2, 'latitude': 45.3})
        """
        # Create timestep entry with NaN for missing signals
        timestep_data = {}
        
        for signal in self.signal_names:
            if signal in signal_values:
                # Signal arrived - use its value
                value = signal_values[signal]
                timestep_data[signal] = value
                # Update last known value
                self.last_known_values[signal] = value
            else:
                # Signal missing - mark as NaN (will forward-fill later)
                timestep_data[signal] = np.nan
        
        # Add timestamp
        timestep_data['timestamp'] = timestamp
        
        # Append to queue (auto-removes oldest if at capacity)
        self.queue.append(timestep_data)
        self.total_enqueued += 1
    
    
    def dequeue(self) -> Optional[Dict[str, float]]:
        """
        Remove and return oldest timestep from queue.
        
        Returns:
            Dictionary with signal values and timestamp, or None if queue empty
        """
        if len(self.queue) == 0:
            return None
        
        self.total_dequeued += 1
        return self.queue.popleft()
    
    
    def _apply_forward_fill_to_queue(self) -> None:
        """
        Fill missing values (NaN) with forward-fill strategy.
        Uses CORRECT chronological iteration (past → future).
        
        CRITICAL: Iterates forward through queue, tracking last_seen values
        to avoid temporal leak (using future values to fill past).
        
        Note: For production use, consider using ForwardFillProcessor.fill_matrix()
        which has the same correct logic.
        """
        # Local tracker for last seen values (NOT global self.last_known_values!)
        last_seen = {signal: None for signal in self.signal_names}
        
        # Iterate chronologically forward through queue
        for timestep_data in self.queue:
            for signal in self.signal_names:
                value = timestep_data[signal]
                
                if not np.isnan(value):
                    # Update local tracker with this value
                    last_seen[signal] = value
                elif last_seen[signal] is not None:
                    # Fill with PAST value (from local tracker)
                    timestep_data[signal] = last_seen[signal]
                # If last_seen is None, leave as NaN (no previous value yet)
    
    
    def get_window(self, size: int = 50, apply_forward_fill: bool = True) -> Optional[np.ndarray]:
        """
        Extract most recent window of specified size.
        
        Args:
            size: Number of timesteps to extract (default: 50 for CNN)
            apply_forward_fill: Whether to fill NaN before extraction (default: True)
        
        Returns:
            numpy array of shape (num_signals, size) or None if not enough data
            Rows are in OPTIMAL ORDER (from signal_order.txt)
        
        Example:
            window = queue.get_window(size=50)
            # Returns shape (15, 50) - 15 signals × 50 timesteps
        """
        if len(self.queue) < size:
            return None  # Not enough data yet
        
        # Apply forward-fill if requested
        if apply_forward_fill:
            self._apply_forward_fill_to_queue()
        
        # Extract last 'size' timesteps
        recent_data = list(self.queue)[-size:]
        
        # Build matrix: rows = signals (in order), cols = timesteps
        matrix = np.zeros((self.num_signals, size))
        
        for col_idx, timestep_data in enumerate(recent_data):
            for row_idx, signal in enumerate(self.signal_names):
                matrix[row_idx, col_idx] = timestep_data[signal]
        
        return matrix
    
    
    def get_all_windows(self, window_size: int = 50, stride: int = 1) -> List[np.ndarray]:
        """
        Extract ALL possible windows with given stride.
        
        Args:
            window_size: Size of each window (default: 50)
            stride: Step between windows (default: 1 = sliding by 1 timestep)
        
        Returns:
            List of numpy arrays, each of shape (num_signals, window_size)
        
        Example:
            If queue has 1000 timesteps, window_size=50, stride=1:
            → Returns 951 windows [(0:50), (1:51), ..., (950:1000)]
        """
        if len(self.queue) < window_size:
            return []
        
        # Apply forward-fill
        self.forward_fill()
        
        windows = []
        num_windows = (len(self.queue) - window_size) // stride + 1
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            # Extract window
            window_data = list(self.queue)[start_idx:end_idx]
            
            # Build matrix
            matrix = np.zeros((self.num_signals, window_size))
            for col_idx, timestep_data in enumerate(window_data):
                for row_idx, signal in enumerate(self.signal_names):
                    matrix[row_idx, col_idx] = timestep_data[signal]
            
            windows.append(matrix)
        
        return windows
    
    
    def get_stats(self) -> Dict:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue metrics
        """
        return {
            'current_size': len(self.queue),
            'capacity': self.capacity,
            'utilization': len(self.queue) / self.capacity if self.capacity > 0 else 0,
            'total_enqueued': self.total_enqueued,
            'total_dequeued': self.total_dequeued,
            'num_signals': self.num_signals,
            'signal_names': self.signal_names,
            'last_known_values': {k: v for k, v in self.last_known_values.items() if v is not None}
        }
    
    
    def clear(self) -> None:
        """
        Clear all data from queue.
        """
        self.queue.clear()
        self.last_known_values = {signal: None for signal in self.signal_names}
    
    
    def __len__(self) -> int:
        """Return current queue size."""
        return len(self.queue)
    
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FIFOQueue(signals={self.num_signals}, size={len(self.queue)}/{self.capacity})"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_signal_order(filepath: str) -> List[str]:
    """
    Load optimal signal order from file.
    
    Args:
        filepath: Path to signal_order.txt
    
    Returns:
        List of signal names in optimal order
    """
    with open(filepath, 'r') as f:
        signals = [line.strip() for line in f.readlines() if line.strip()]
    return signals


def create_queue_from_csv(csv_path: str, signal_order_path: str, 
                          capacity: int = 1000) -> FIFOQueue:
    """
    Create and populate FIFO queue from decoded CAN CSV.
    
    Args:
        csv_path: Path to decoded_brute_frames.csv
        signal_order_path: Path to signal_order.txt
        capacity: Queue capacity
    
    Returns:
        Populated FIFOQueue instance
    """
    # Load optimal signal order
    signal_names = load_signal_order(signal_order_path)
    
    # Create queue
    queue = FIFOQueue(signal_names=signal_names, capacity=capacity)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Group by timestamp and enqueue
    for timestamp, group in df.groupby('timestamp'):
        signal_values = {}
        for _, row in group.iterrows():
            # Assuming CSV has columns: timestamp, signal_name, value
            if 'signal_name' in row and 'value' in row:
                signal_values[row['signal_name']] = row['value']
        
        queue.enqueue(timestamp, signal_values)
    
    return queue


# ==============================================================================
# TESTING / DEMO
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FIFO Queue - Testing Module")
    print("="*80)
    print()
    
    # Test 1: Basic queue operations
    print("TEST 1: Basic Queue Operations")
    print("-" * 80)
    
    signal_order = ['pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle']
    queue = FIFOQueue(signal_names=signal_order, capacity=10)
    
    print(f"Created queue: {queue}")
    print()
    
    # Enqueue some data
    print("Enqueuing 5 timesteps...")
    for t in range(5):
        signals = {
            'pitch': 2.5 + t * 0.1,
            'depth': 50.0 + t * 0.5,
            'wind_speed': 10.0 + t * 0.2,
            # Intentionally missing: yaw, wind_angle
        }
        queue.enqueue(timestamp=float(t), signal_values=signals)
    
    print(f"Queue after enqueue: {queue}")
    print(f"Stats: {queue.get_stats()}")
    print()
    
    # Test 2: Forward fill
    print("TEST 2: Forward Fill")
    print("-" * 80)
    queue.forward_fill()
    print("Forward-fill applied")
    print()
    
    # Test 3: Extract window
    print("TEST 3: Extract Window")
    print("-" * 80)
    window = queue.get_window(size=5)
    if window is not None:
        print(f"Window shape: {window.shape}")
        print(f"Window data:\n{window}")
    else:
        print("Not enough data for window")
    print()
    
    print("="*80)
    print("FIFO Queue - Tests Complete!")
    print("="*80)
