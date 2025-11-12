"""
Multi-scale view builder
Creates V_Tx views from FIFO queue snapshots
Based on CANShield Phase 1
"""
import numpy as np
from typing import List, Tuple


class ViewBuilder:
    """Build multi-scale temporal views from FIFO queue"""
    
    def __init__(self, sampling_periods: List[int] = [1, 5, 10, 20, 50]):
        """
        Args:
            sampling_periods: List of sampling periods Tx
        """
        self.sampling_periods = sampling_periods
    
    def build_view(self, queue_snapshot: np.ndarray, period: int) -> np.ndarray:
        """
        Build view V_Tx by sampling every Tx time steps
        
        Args:
            queue_snapshot: 2D array from FIFO queue (q, n_signals)
            period: Sampling period Tx
        
        Returns:
            View V_Tx as 2D array (q//Tx, n_signals)
        """
        # Sample every Tx rows
        view = queue_snapshot[::period, :]
        
        return view
    
    def build_all_views(self, queue_snapshot: np.ndarray) -> dict:
        """
        Build all multi-scale views
        
        Args:
            queue_snapshot: 2D array from FIFO queue
        
        Returns:
            Dictionary {period: view_array}
        """
        views = {}
        
        for period in self.sampling_periods:
            views[period] = self.build_view(queue_snapshot, period)
        
        return views
    
    def get_view_shape(self, queue_size: int, period: int, n_signals: int) -> Tuple[int, int]:
        """
        Get expected view shape
        
        Args:
            queue_size: FIFO queue size (q)
            period: Sampling period Tx
            n_signals: Number of signals
        
        Returns:
            Tuple (height, width)
        """
        height = queue_size // period
        width = n_signals
        return (height, width)


if __name__ == "__main__":
    builder = ViewBuilder()
    print(f"ViewBuilder ready with periods: {builder.sampling_periods}")
