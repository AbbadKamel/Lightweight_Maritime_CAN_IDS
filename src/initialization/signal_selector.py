"""
Signal selector - Select critical signals from N2K data
Based on CANShield initialization phase
"""
import pandas as pd
import numpy as np
from typing import List, Dict


class SignalSelector:
    """Select critical signals from NMEA 2000 data"""
    
    def __init__(self, min_variance_threshold: float = 0.01):
        """
        Args:
            min_variance_threshold: Minimum variance to consider signal critical
        """
        self.min_variance_threshold = min_variance_threshold
        self.selected_signals = []
    
    def select(self, data: pd.DataFrame) -> List[str]:
        """
        Select critical signals based on variance
        
        Args:
            data: DataFrame with signal columns
        
        Returns:
            List of selected signal names
        """
        variances = data.var()
        
        # Select signals with variance > threshold
        critical_signals = variances[variances > self.min_variance_threshold].index.tolist()
        
        self.selected_signals = critical_signals
        print(f"Selected {len(critical_signals)} critical signals out of {len(data.columns)}")
        
        return critical_signals
    
    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to keep only selected signals"""
        if not self.selected_signals:
            raise ValueError("No signals selected. Run select() first.")
        
        return data[self.selected_signals]


if __name__ == "__main__":
    print("SignalSelector ready")
