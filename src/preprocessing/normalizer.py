"""
Min-Max normalizer for signal values
Based on CANShield Phase 1
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Union
import pickle


class Normalizer:
    """Min-Max normalization for signal values"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Fit normalizer on training data
        
        Args:
            data: Training data (2D array or DataFrame)
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        self.scaler.fit(data)
        self.is_fitted = True
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted scaler
        
        Args:
            data: Data to normalize
        
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        return self.scaler.transform(data)
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            data: Data to fit and normalize
        
        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data
        
        Returns:
            Original scale data
        """
        return self.scaler.inverse_transform(data)
    
    def save(self, filepath: str):
        """Save scaler to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, filepath: str):
        """Load scaler from file"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
    
    def get_min_max_values(self) -> pd.DataFrame:
        """
        Get min/max values for each feature
        
        Returns:
            DataFrame with min and max values
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted.")
        
        return pd.DataFrame({
            'min': self.scaler.data_min_,
            'max': self.scaler.data_max_
        })


if __name__ == "__main__":
    normalizer = Normalizer()
    print("Normalizer ready")
