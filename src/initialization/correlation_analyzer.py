"""
Correlation analyzer - Compute Pearson correlation and hierarchical clustering
Based on CANShield initialization phase
"""
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


class CorrelationAnalyzer:
    """Analyze signal correlations and perform hierarchical clustering"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.linkage_matrix = None
        self.signal_order = None
    
    def compute_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Pearson correlation matrix
        
        Args:
            data: DataFrame with signal columns
        
        Returns:
            Correlation matrix
        """
        self.correlation_matrix = data.corr(method='pearson')
        return self.correlation_matrix
    
    def hierarchical_clustering(self, method: str = 'average') -> np.ndarray:
        """
        Perform hierarchical clustering on correlation matrix
        
        Args:
            method: Linkage method ('average', 'single', 'complete', 'ward')
        
        Returns:
            Linkage matrix
        """
        if self.correlation_matrix is None:
            raise ValueError("Compute correlation first using compute_correlation()")
        
        # Convert correlation to distance: d = 1 - |corr|
        distance_matrix = 1 - np.abs(self.correlation_matrix)
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Hierarchical clustering
        self.linkage_matrix = hierarchy.linkage(condensed_dist, method=method)
        
        return self.linkage_matrix
    
    def get_signal_order(self) -> List[str]:
        """
        Get optimal signal ordering from dendrogram
        
        Returns:
            List of signal names in optimal order
        """
        if self.linkage_matrix is None:
            raise ValueError("Perform clustering first using hierarchical_clustering()")
        
        # Get dendrogram leaf order
        dendro = hierarchy.dendrogram(self.linkage_matrix, no_plot=True)
        leaf_order = dendro['leaves']
        
        # Reorder signals
        signal_names = self.correlation_matrix.columns.tolist()
        self.signal_order = [signal_names[i] for i in leaf_order]
        
        return self.signal_order
    
    def plot_correlation_matrix(self, save_path: str = None):
        """Plot correlation matrix heatmap"""
        if self.correlation_matrix is None:
            raise ValueError("Compute correlation first")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlation_matrix, cmap='coolwarm', center=0, 
                    vmin=-1, vmax=1, square=True)
        plt.title("Signal Correlation Matrix")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dendrogram(self, save_path: str = None):
        """Plot hierarchical clustering dendrogram"""
        if self.linkage_matrix is None:
            raise ValueError("Perform clustering first")
        
        plt.figure(figsize=(15, 7))
        hierarchy.dendrogram(self.linkage_matrix, labels=self.correlation_matrix.columns)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Signals")
        plt.ylabel("Distance")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    print("CorrelationAnalyzer ready")
