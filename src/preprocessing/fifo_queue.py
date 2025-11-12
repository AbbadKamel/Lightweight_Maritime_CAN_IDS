"""
FIFO Queue management for CAN messages
Based on CANShield Phase 1
"""
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple


class FIFOQueue:
    """FIFO queue for CAN signal history (Q in CANShield paper)"""
    
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum queue size (q in paper, default 1000)
        """
        self.max_size = max_size
        self.queue = deque(maxlen=max_size)
        self.signal_names = None
    
    def push(self, signals: np.ndarray):
        """
        Push new signal vector to queue
        
        Args:
            signals: 1D array of signal values
        """
        self.queue.append(signals)
    
    def get_snapshot(self) -> np.ndarray:
        """
        Get current queue snapshot as 2D array
        
        Returns:
            2D array of shape (queue_size, n_signals)
        """
        if len(self.queue) == 0:
            return np.array([])
        
        return np.array(list(self.queue))
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return len(self.queue) >= self.max_size
    
    def clear(self):
        """Clear the queue"""
        self.queue.clear()
    
    def __len__(self):
        return len(self.queue)
    
    def __repr__(self):
        return f"FIFOQueue(size={len(self)}/{self.max_size})"


if __name__ == "__main__":
    queue = FIFOQueue(max_size=1000)
    print(f"FIFOQueue ready: {queue}")
