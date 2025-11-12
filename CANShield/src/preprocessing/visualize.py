"""
Visualization Tools for CANShield Preprocessing
================================================

Purpose:
    - Visualize preprocessing results
    - Validate data quality
    - Compare before/after normalization
    - Inspect multi-scale views
    - Analyze forward-fill behavior

Features:
    - Plot signal time series
    - Visualize multi-scale windows
    - Show normalization effects
    - Display correlation with heatmaps
    - Animate multi-scale views

Author: CANShield Implementation (Maritime Version)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_signal_timeseries(data: np.ndarray,
                           signal_names: List[str],
                           title: str = "Signal Time Series",
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[Union[str, Path]] = None):
    """
    Plot all signals as time series.
    
    Args:
        data: Array of shape (num_signals, num_timesteps)
        signal_names: List of signal names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
    
    Example:
        >>> data = np.random.randn(15, 1000)
        >>> plot_signal_timeseries(data, signal_names, "Raw CAN Signals")
    """
    num_signals = data.shape[0]
    num_cols = 3
    num_rows = (num_signals + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if num_signals > 1 else [axes]
    
    for i, signal in enumerate(signal_names):
        ax = axes[i]
        ax.plot(data[i, :], linewidth=0.8, alpha=0.8)
        ax.set_title(f"{signal}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.nanmean(data[i, :])
        std_val = np.nanstd(data[i, :])
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide extra subplots
    for i in range(num_signals, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def plot_multiscale_comparison(views: Dict[int, np.ndarray],
                               signal_names: List[str],
                               signal_index: int = 0,
                               figsize: Tuple[int, int] = (15, 8),
                               save_path: Optional[Union[str, Path]] = None):
    """
    Compare same signal across different sampling periods.
    
    Args:
        views: Dictionary mapping sampling_period → view array (num_signals, window_size)
        signal_names: List of signal names
        signal_index: Which signal to plot (default: 0 = first signal)
        figsize: Figure size
        save_path: Path to save figure
    
    Example:
        >>> views = {1: view1, 5: view5, 10: view10}
        >>> plot_multiscale_comparison(views, signal_names, signal_index=0)
    """
    signal_name = signal_names[signal_index]
    num_views = len(views)
    
    fig, axes = plt.subplots(num_views, 1, figsize=figsize, sharex=False)
    if num_views == 1:
        axes = [axes]
    
    for idx, (T, view) in enumerate(sorted(views.items())):
        ax = axes[idx]
        signal_data = view[signal_index, :]
        
        ax.plot(signal_data, marker='o', markersize=3, linewidth=1.5, alpha=0.7)
        ax.set_ylabel(f"T={T}\n(every {T} steps)", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(signal_data)
        ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5)
        ax.text(0.98, 0.95, f'μ={mean_val:.3f}',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    axes[-1].set_xlabel("Window Position", fontsize=11)
    fig.suptitle(f"Multi-Scale View Comparison: {signal_name}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def plot_normalization_effect(original: np.ndarray,
                              normalized: np.ndarray,
                              signal_names: List[str],
                              figsize: Tuple[int, int] = (15, 8),
                              save_path: Optional[Union[str, Path]] = None):
    """
    Compare original vs normalized signals.
    
    Args:
        original: Original data (num_signals, num_timesteps)
        normalized: Normalized data (num_signals, num_timesteps)
        signal_names: List of signal names
        figsize: Figure size
        save_path: Path to save figure
    """
    num_signals = min(6, len(signal_names))  # Show first 6 signals
    
    fig, axes = plt.subplots(num_signals, 2, figsize=figsize)
    
    for i in range(num_signals):
        # Original
        axes[i, 0].plot(original[i, :500], linewidth=0.8, color='blue', alpha=0.7)
        axes[i, 0].set_ylabel(signal_names[i], fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=11, fontweight='bold')
        
        # Original statistics
        orig_min, orig_max = original[i, :].min(), original[i, :].max()
        axes[i, 0].text(0.02, 0.98, f'[{orig_min:.2f}, {orig_max:.2f}]',
                       transform=axes[i, 0].transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Normalized
        axes[i, 1].plot(normalized[i, :500], linewidth=0.8, color='green', alpha=0.7)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylim(-0.1, 1.1)
        if i == 0:
            axes[i, 1].set_title("Normalized [0, 1]", fontsize=11, fontweight='bold')
        
        # Normalized statistics
        norm_min, norm_max = normalized[i, :].min(), normalized[i, :].max()
        axes[i, 1].text(0.02, 0.98, f'[{norm_min:.3f}, {norm_max:.3f}]',
                       transform=axes[i, 1].transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    axes[-1, 0].set_xlabel("Timestep (first 500)", fontsize=10)
    axes[-1, 1].set_xlabel("Timestep (first 500)", fontsize=10)
    
    fig.suptitle("Normalization Effect Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def plot_window_as_heatmap(window: np.ndarray,
                           signal_names: List[str],
                           title: str = "CAN Window Heatmap",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[Union[str, Path]] = None):
    """
    Visualize a single window as a heatmap.
    
    Args:
        window: Window array (num_signals, window_size) or (num_signals, window_size, 1)
        signal_names: List of signal names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Example:
        >>> window = normalized_windows[0]  # First window
        >>> plot_window_as_heatmap(window, signal_names, "Example Window")
    """
    # Remove channel dimension if present
    if window.ndim == 3:
        window = window.squeeze(-1)
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(window, 
                yticklabels=signal_names,
                cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Value'},
                linewidths=0.5,
                linecolor='gray',
                vmin=0, vmax=1)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Timestep", fontsize=11)
    plt.ylabel("Signal", fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def plot_multiple_windows(windows: np.ndarray,
                         signal_names: List[str],
                         num_windows: int = 4,
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[Union[str, Path]] = None):
    """
    Plot multiple windows side by side.
    
    Args:
        windows: Array of shape (num_windows, num_signals, window_size, 1)
        signal_names: List of signal names
        num_windows: Number of windows to show
        figsize: Figure size
        save_path: Path to save figure
    """
    num_windows = min(num_windows, windows.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_windows):
        window = windows[i].squeeze(-1)  # Remove channel dimension
        
        sns.heatmap(window,
                   yticklabels=signal_names if i % 2 == 0 else False,
                   cmap='RdYlGn',
                   cbar=True,
                   ax=axes[i],
                   vmin=0, vmax=1)
        
        axes[i].set_title(f"Window {i+1}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Timestep")
        if i % 2 == 0:
            axes[i].set_ylabel("Signal")
    
    fig.suptitle("Sample Windows from Dataset", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def plot_forward_fill_demonstration(data_with_nan: np.ndarray,
                                    data_filled: np.ndarray,
                                    signal_names: List[str],
                                    signal_index: int = 0,
                                    figsize: Tuple[int, int] = (14, 5),
                                    save_path: Optional[Union[str, Path]] = None):
    """
    Demonstrate forward-fill effect on data with NaN.
    
    Args:
        data_with_nan: Original data with NaN (num_signals, num_timesteps)
        data_filled: Forward-filled data (num_signals, num_timesteps)
        signal_names: List of signal names
        signal_index: Which signal to show
        figsize: Figure size
        save_path: Path to save figure
    """
    signal_name = signal_names[signal_index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Before forward-fill
    ax1.plot(data_with_nan[signal_index, :200], marker='o', markersize=4,
             linewidth=1.5, label='Valid data')
    ax1.set_title(f"Before Forward-Fill: {signal_name}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Count NaN
    nan_count = np.isnan(data_with_nan[signal_index, :200]).sum()
    ax1.text(0.02, 0.98, f'NaN count: {nan_count}',
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # After forward-fill
    ax2.plot(data_filled[signal_index, :200], marker='o', markersize=4,
             linewidth=1.5, color='green', label='Forward-filled')
    ax2.set_title(f"After Forward-Fill: {signal_name}", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Count NaN
    nan_count_after = np.isnan(data_filled[signal_index, :200]).sum()
    ax2.text(0.02, 0.98, f'NaN count: {nan_count_after}',
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def plot_distribution_comparison(data_dict: Dict[str, np.ndarray],
                                 signal_names: List[str],
                                 signal_index: int = 0,
                                 figsize: Tuple[int, int] = (12, 6),
                                 save_path: Optional[Union[str, Path]] = None):
    """
    Compare signal distributions across different processing stages.
    
    Args:
        data_dict: Dictionary mapping label → data array
        signal_names: List of signal names
        signal_index: Which signal to analyze
        figsize: Figure size
        save_path: Path to save figure
    
    Example:
        >>> data_dict = {
        ...     'Raw': raw_data,
        ...     'Forward-filled': filled_data,
        ...     'Normalized': normalized_data
        ... }
        >>> plot_distribution_comparison(data_dict, signal_names, 0)
    """
    signal_name = signal_names[signal_index]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    for label, data in data_dict.items():
        signal_data = data[signal_index, :]
        # Remove NaN
        valid_data = signal_data[~np.isnan(signal_data)]
        axes[0].hist(valid_data, bins=50, alpha=0.6, label=label, density=True)
    
    axes[0].set_title(f"Distribution: {signal_name}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_for_box = []
    labels_for_box = []
    for label, data in data_dict.items():
        signal_data = data[signal_index, :]
        valid_data = signal_data[~np.isnan(signal_data)]
        data_for_box.append(valid_data)
        labels_for_box.append(label)
    
    axes[1].boxplot(data_for_box, labels=labels_for_box)
    axes[1].set_title(f"Box Plot: {signal_name}", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()


def create_preprocessing_report(csv_path: Union[str, Path],
                                signal_order_path: Union[str, Path],
                                output_dir: Union[str, Path],
                                num_samples: int = 5000):
    """
    Generate comprehensive visualization report of preprocessing pipeline.
    
    Args:
        csv_path: Path to CSV data
        signal_order_path: Path to signal_order.txt
        output_dir: Directory to save visualizations
        num_samples: Number of samples to process (for speed)
    
    Creates:
        - signal_timeseries.png
        - normalization_effect.png
        - sample_windows.png
        - multiscale_comparison.png
        - distribution_analysis.png
    """
    from .data_loader import CANDataLoader, load_signal_order
    from .multi_scale import MultiScaleGenerator
    from .normalization import SignalNormalizer
    
    print("=" * 80)
    print("PREPROCESSING VISUALIZATION REPORT")
    print("=" * 80)
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load signal order
    signal_names = load_signal_order(signal_order_path)
    print(f"\n✅ Loaded {len(signal_names)} signals")
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df.head(num_samples)  # Limit for speed
    
    # Extract signals
    data_raw = df[signal_names].values.T
    print(f"✅ Loaded data: {data_raw.shape}")
    
    # 1. Plot raw signals
    print("\n[1/6] Plotting raw signal time series...")
    plot_signal_timeseries(
        data_raw,
        signal_names,
        "Raw CAN Signal Time Series",
        save_path=output_dir / 'signal_timeseries.png'
    )
    
    # 2. Apply forward-fill
    print("\n[2/6] Applying forward-fill...")
    data_filled = pd.DataFrame(data_raw.T, columns=signal_names).ffill().bfill().values.T
    
    # 3. Normalize
    print("\n[3/6] Normalizing data...")
    normalizer = SignalNormalizer(signal_names)
    data_normalized = normalizer.fit_transform(data_filled)
    
    plot_normalization_effect(
        data_filled[:, :1000],
        data_normalized[:, :1000],
        signal_names,
        save_path=output_dir / 'normalization_effect.png'
    )
    
    # 4. Generate multi-scale views
    print("\n[4/6] Generating multi-scale views...")
    generator = MultiScaleGenerator([1, 5, 10, 20, 50], window_size=50)
    views = generator.generate_views(data_normalized)
    
    plot_multiscale_comparison(
        views,
        signal_names,
        signal_index=0,
        save_path=output_dir / 'multiscale_comparison.png'
    )
    
    # 5. Generate windows
    print("\n[5/6] Generating training windows...")
    windows_dict = generator.generate_sliding_windows(data_normalized, stride=50)
    
    # Reshape for visualization
    windows_T1 = windows_dict[1][..., np.newaxis]  # Add channel dimension
    
    plot_multiple_windows(
        windows_T1[:4],
        signal_names,
        num_windows=4,
        save_path=output_dir / 'sample_windows.png'
    )
    
    # 6. Distribution analysis
    print("\n[6/6] Analyzing distributions...")
    data_dict = {
        'Raw': data_raw,
        'Normalized': data_normalized
    }
    
    plot_distribution_comparison(
        data_dict,
        signal_names,
        signal_index=0,
        save_path=output_dir / 'distribution_analysis.png'
    )
    
    print("\n" + "=" * 80)
    print("✅ REPORT COMPLETE!")
    print(f"📁 Visualizations saved to: {output_dir}")
    print("=" * 80)


# ==============================================================================
# QUICK DEMO
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Visualization Tools - Demo")
    print("=" * 80)
    
    # Create synthetic data
    signal_names = ['pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle']
    np.random.seed(42)
    
    # 1. Raw signal time series
    print("\n[Demo 1] Raw Signal Time Series")
    data_raw = np.array([
        np.random.randn(500) * 5 + 0,
        np.random.randn(500) * 10 + 50,
        np.random.randn(500) * 3 + 10,
        np.random.randn(500) * 10 + 90,
        np.random.randn(500) * 30 + 180,
    ])
    
    plot_signal_timeseries(data_raw, signal_names, "Demo: Raw CAN Signals")
    
    # 2. Normalization effect
    print("\n[Demo 2] Normalization Effect")
    data_normalized = (data_raw - data_raw.min(axis=1, keepdims=True)) / \
                     (data_raw.max(axis=1, keepdims=True) - data_raw.min(axis=1, keepdims=True))
    
    plot_normalization_effect(data_raw, data_normalized, signal_names)
    
    # 3. Multi-scale views
    print("\n[Demo 3] Multi-Scale Views")
    from multi_scale import MultiScaleGenerator
    
    generator = MultiScaleGenerator([1, 5, 10], window_size=50)
    views = generator.generate_views(data_normalized)
    
    plot_multiscale_comparison(views, signal_names, signal_index=0)
    
    # 4. Window heatmap
    print("\n[Demo 4] Window Heatmap")
    window = views[1]  # T=1 view
    plot_window_as_heatmap(window, signal_names, "Demo: Single Window")
    
    # 5. Distribution comparison
    print("\n[Demo 5] Distribution Comparison")
    data_dict = {
        'Raw': data_raw,
        'Normalized': data_normalized
    }
    plot_distribution_comparison(data_dict, signal_names, 0)
    
    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)
