"""
Signal reorderer - Reorder signals based on hierarchical clustering
Based on CANShield initialization phase
"""
import pandas as pd
from typing import List


class SignalReorderer:
    """Reorder signals based on correlation clustering"""
    
    def __init__(self, signal_order: List[str]):
        """
        Args:
            signal_order: Optimal signal order from hierarchical clustering
        """
        self.signal_order = signal_order
    
    def reorder(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns based on optimal signal order
        
        Args:
            data: DataFrame with signal columns
        
        Returns:
            Reordered DataFrame
        """
        # Keep only signals in the order list
        available_signals = [s for s in self.signal_order if s in data.columns]
        
        return data[available_signals]
    
    def save_order(self, filepath: str):
        """Save signal order to file"""
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.signal_order))
    
    @staticmethod
    def load_order(filepath: str) -> List[str]:
        """Load signal order from file"""
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines()]


if __name__ == "__main__":
    print("SignalReorderer ready")
