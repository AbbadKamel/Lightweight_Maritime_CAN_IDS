"""
Forward-Fill Strategy for Missing CAN Signal Values
====================================================

Purpose:
    - Handle asynchronous signal arrival
    - Fill missing values with last-known-good values
    - Ensure no NaN in data matrices for CNN

Strategy:
    - Track last valid value for each signal
    - When signal missing → use last known value
    - Works with FIFO Queue for real-time processing

Author: CANShield Project
Date: 2025-11-07
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class ForwardFillProcessor:
    """
    Manages forward-fill strategy for CAN signals.
    
    Maintains last-known-good values and fills missing data.
    """
    
    def __init__(self, signal_names: List[str]):
        """
        Initialize forward-fill processor.
        
        Args:
            signal_names: List of signal names to track
        """
        self.signal_names = signal_names
        self.last_known_values = {signal: None for signal in signal_names}
        self.fill_count = {signal: 0 for signal in signal_names}  # Track fills per signal
    
    
    def update(self, signal_values: Dict[str, float]) -> Dict[str, float]:
        """
        Update last known values and fill missing signals.
        
        Args:
            signal_values: Dictionary with available signal values
        
        Returns:
            Complete dictionary with all signals (filled if missing)
        """
        filled_values = {}
        
        for signal in self.signal_names:
            if signal in signal_values and not np.isnan(signal_values[signal]):
                # Signal present and valid → update last known
                self.last_known_values[signal] = signal_values[signal]
                filled_values[signal] = signal_values[signal]
            else:
                # Signal missing → forward-fill
                if self.last_known_values[signal] is not None:
                    filled_values[signal] = self.last_known_values[signal]
                    self.fill_count[signal] += 1
                else:
                    # No previous value → keep NaN (will be handled by pandas bfill in training)
                    filled_values[signal] = np.nan
        
        return filled_values
    
    
    def fill_dataframe(self, df: pd.DataFrame, 
                      timestamp_col: str = 'timestamp',
                      signal_col: str = 'signal_name',
                      value_col: str = 'value') -> pd.DataFrame:
        """
        Apply forward-fill to entire DataFrame.
        
        Args:
            df: DataFrame with CAN messages
            timestamp_col: Name of timestamp column
            signal_col: Name of signal name column
            value_col: Name of value column
        
        Returns:
            DataFrame with all signals at each timestamp (forward-filled)
        """
        # Get unique timestamps
        timestamps = sorted(df[timestamp_col].unique())
        
        # Build filled data
        filled_data = []
        
        for ts in timestamps:
            # Get signals at this timestamp
            ts_data = df[df[timestamp_col] == ts]
            signal_values = dict(zip(ts_data[signal_col], ts_data[value_col]))
            
            # Fill missing signals
            filled_signals = self.update(signal_values)
            
            # Add to result
            for signal, value in filled_signals.items():
                filled_data.append({
                    timestamp_col: ts,
                    signal_col: signal,
                    value_col: value,
                    'is_filled': signal not in signal_values
                })
        
        return pd.DataFrame(filled_data)
    
    
    def fill_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply forward-fill to matrix (in-place along time axis).
        
        Args:
            matrix: Shape (num_signals, num_timesteps) with possible NaN
        
        Returns:
            Filled matrix (same shape, no NaN)
        """
        num_signals, num_timesteps = matrix.shape
        filled = matrix.copy()
        
        for signal_idx in range(num_signals):
            last_valid = None
            
            for time_idx in range(num_timesteps):
                value = filled[signal_idx, time_idx]
                
                if not np.isnan(value):
                    last_valid = value
                else:
                    # Forward-fill
                    if last_valid is not None:
                        filled[signal_idx, time_idx] = last_valid
                    else:
                        filled[signal_idx, time_idx] = 0.0  # Edge case
        
        return filled
    
    
    def get_statistics(self) -> Dict:
        """
        Get forward-fill statistics.
        
        Returns:
            Dictionary with fill counts and coverage
        """
        total_fills = sum(self.fill_count.values())
        
        return {
            'total_fills': total_fills,
            'fills_per_signal': self.fill_count,
            'signals_with_data': sum(1 for v in self.last_known_values.values() if v is not None),
            'last_known_values': {k: v for k, v in self.last_known_values.items() if v is not None}
        }
    
    
    def reset(self) -> None:
        """Reset all last known values and statistics."""
        self.last_known_values = {signal: None for signal in self.signal_names}
        self.fill_count = {signal: 0 for signal in self.signal_names}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def apply_forward_fill_to_csv(input_csv: str, output_csv: str, 
                              signal_order_path: str) -> pd.DataFrame:
    """
    Load CSV, apply forward-fill, save result.
    
    Args:
        input_csv: Path to decoded_brute_frames.csv
        output_csv: Path to save filled data
        signal_order_path: Path to signal_order.txt
    
    Returns:
        Filled DataFrame
    """
    # Load signal order
    with open(signal_order_path, 'r') as f:
        signal_names = [line.strip() for line in f if line.strip()]
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Create processor
    processor = ForwardFillProcessor(signal_names)
    
    # Fill data
    filled_df = processor.fill_dataframe(df)
    
    # Save
    filled_df.to_csv(output_csv, index=False)
    
    # Print statistics
    stats = processor.get_statistics()
    print(f"Forward-fill complete:")
    print(f"  Total fills: {stats['total_fills']}")
    print(f"  Signals with data: {stats['signals_with_data']}/{len(signal_names)}")
    
    return filled_df


# ==============================================================================
# TESTING / DEMO
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Forward-Fill Processor - Testing Module")
    print("="*80)
    print()
    
    # Test 1: Basic forward-fill
    print("TEST 1: Basic Forward-Fill")
    print("-" * 80)
    
    signals = ['wind_speed', 'latitude', 'depth']
    processor = ForwardFillProcessor(signals)
    
    # Timestep 1: All signals present
    print("Timestep 1: All signals present")
    values1 = {'wind_speed': 10.0, 'latitude': 45.0, 'depth': 50.0}
    filled1 = processor.update(values1)
    print(f"  Input:  {values1}")
    print(f"  Output: {filled1}")
    print()
    
    # Timestep 2: Missing latitude
    print("Timestep 2: Missing latitude")
    values2 = {'wind_speed': 10.5, 'depth': 51.0}
    filled2 = processor.update(values2)
    print(f"  Input:  {values2}")
    print(f"  Output: {filled2}")
    print(f"  → latitude filled with: {filled2['latitude']} (from timestep 1)")
    print()
    
    # Timestep 3: Missing wind_speed and depth
    print("Timestep 3: Missing wind_speed and depth")
    values3 = {'latitude': 45.5}
    filled3 = processor.update(values3)
    print(f"  Input:  {values3}")
    print(f"  Output: {filled3}")
    print(f"  → wind_speed filled with: {filled3['wind_speed']} (from timestep 2)")
    print(f"  → depth filled with: {filled3['depth']} (from timestep 2)")
    print()
    
    # Statistics
    print("Forward-Fill Statistics:")
    stats = processor.get_statistics()
    print(f"  Total fills: {stats['total_fills']}")
    print(f"  Fills per signal: {stats['fills_per_signal']}")
    print()
    
    # Test 2: Matrix forward-fill
    print("TEST 2: Matrix Forward-Fill")
    print("-" * 80)
    
    # Create matrix with NaN
    matrix = np.array([
        [1.0, 2.0, np.nan, 4.0, 5.0],
        [10.0, np.nan, np.nan, 13.0, 14.0],
        [np.nan, 22.0, 23.0, np.nan, 25.0]
    ])
    
    print("Original matrix (with NaN):")
    print(matrix)
    print()
    
    processor2 = ForwardFillProcessor(['sig1', 'sig2', 'sig3'])
    filled_matrix = processor2.fill_matrix(matrix)
    
    print("Filled matrix:")
    print(filled_matrix)
    print()
    
    print("="*80)
    print("Forward-Fill Processor - Tests Complete!")
    print("="*80)
