#!/usr/bin/env python3
"""
PHASE 2: Training Results Visualization
========================================

This script visualizes all training results:
- Loss curves for all 5 autoencoders
- Validation loss comparison
- Training summary statistics
- Threshold analysis

Author: PhD Project - Lightweight IA
Date: November 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / 'results' / 'training'
HISTORIES_DIR = TRAINING_DIR / 'histories'
THRESHOLDS_DIR = TRAINING_DIR / 'thresholds'
MODELS_DIR = TRAINING_DIR / 'models'

TIME_SCALES = [1, 5, 10, 20, 50]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_history(time_scale):
    """Load training history for a given time scale."""
    history_path = HISTORIES_DIR / f'history_T{time_scale}.json'
    with open(history_path, 'r') as f:
        return json.load(f)

def load_thresholds(time_scale):
    """Load thresholds for a given time scale."""
    threshold_path = THRESHOLDS_DIR / f'thresholds_T{time_scale}.json'
    with open(threshold_path, 'r') as f:
        return json.load(f)

def print_training_summary():
    """Print comprehensive training summary."""
    print("\n" + "="*80)
    print("PHASE 2 TRAINING RESULTS SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for T in TIME_SCALES:
        # Load history
        history = load_history(T)
        
        # Find best epoch (lowest val_loss)
        val_losses = history['val_loss']
        best_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_epoch]
        
        # Final epoch metrics
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        final_mae = history['mae'][-1]
        final_val_mae = history['val_mae'][-1]
        
        # Total epochs
        total_epochs = len(history['loss'])
        
        # Load thresholds
        thresholds = load_thresholds(T)
        global_p99 = thresholds['global']['p99']
        
        summary_data.append({
            'T': T,
            'Epochs': total_epochs,
            'Best Epoch': best_epoch + 1,
            'Best Val Loss': best_val_loss,
            'Final Loss': final_loss,
            'Final Val Loss': final_val_loss,
            'Final MAE': final_mae,
            'Final Val MAE': final_val_mae,
            'Threshold p99': global_p99
        })
        
        # Print detailed info
        print(f"\n{'─'*80}")
        print(f"TIME SCALE T={T}")
        print(f"{'─'*80}")
        print(f"  Training:")
        print(f"    Total epochs:        {total_epochs}")
        print(f"    Best epoch:          {best_epoch + 1}")
        print(f"    Early stopped at:    {total_epochs}")
        print(f"  ")
        print(f"  Loss Metrics:")
        print(f"    Best val_loss:       {best_val_loss:.6f}")
        print(f"    Final loss:          {final_loss:.6f}")
        print(f"    Final val_loss:      {final_val_loss:.6f}")
        print(f"  ")
        print(f"  MAE Metrics:")
        print(f"    Final MAE:           {final_mae:.6f}")
        print(f"    Final val_MAE:       {final_val_mae:.6f}")
        print(f"  ")
        print(f"  Threshold:")
        print(f"    Global p99:          {global_p99:.6f}")
        print(f"  ")
        print(f"  Quality Check:")
        
        # Quality indicators
        overfitting = final_val_loss / final_loss if final_loss > 0 else 0
        if overfitting < 1.5:
            print(f"    ✓ No overfitting     (val/train ratio: {overfitting:.2f})")
        else:
            print(f"    ⚠ Possible overfitting (val/train ratio: {overfitting:.2f})")
        
        if final_val_loss < 0.03:
            print(f"    ✓ Excellent val_loss ({final_val_loss:.6f} < 0.03)")
        elif final_val_loss < 0.05:
            print(f"    ✓ Good val_loss      ({final_val_loss:.6f} < 0.05)")
        else:
            print(f"    ⚠ High val_loss      ({final_val_loss:.6f})")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    
    df = pd.DataFrame(summary_data)
    print(f"\n{df.to_string(index=False)}")
    
    return summary_data

def plot_all_training_curves():
    """Plot training curves for all time scales."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Training Results for All Time Scales', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for idx, T in enumerate(TIME_SCALES):
        ax = axes[idx]
        
        # Load history
        history = load_history(T)
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot loss
        ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss'])
        best_val_loss = history['val_loss'][best_epoch]
        ax.plot(best_epoch + 1, best_val_loss, 'g*', markersize=15, 
                label=f'Best (epoch {best_epoch + 1})')
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (MSE)', fontsize=11)
        ax.set_title(f'T={T} - {len(epochs)} epochs', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text box with final metrics
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        textstr = f'Final Loss: {final_loss:.6f}\nFinal Val: {final_val_loss:.6f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide the 6th subplot (we only have 5)
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = TRAINING_DIR / 'training_curves_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved: {output_path}")
    
    return fig

def plot_mae_curves():
    """Plot MAE curves for all time scales."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('MAE (Mean Absolute Error) for All Time Scales', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, T in enumerate(TIME_SCALES):
        ax = axes[idx]
        
        # Load history
        history = load_history(T)
        epochs = range(1, len(history['mae']) + 1)
        
        # Plot MAE
        ax.plot(epochs, history['mae'], 'b-', label='Training MAE', linewidth=2)
        ax.plot(epochs, history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss'])  # Use val_loss for best epoch
        best_val_mae = history['val_mae'][best_epoch]
        ax.plot(best_epoch + 1, best_val_mae, 'g*', markersize=15,
                label=f'Best (epoch {best_epoch + 1})')
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f'T={T} - MAE Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text box
        final_mae = history['mae'][-1]
        final_val_mae = history['val_mae'][-1]
        textstr = f'Final MAE: {final_mae:.6f}\nFinal Val MAE: {final_val_mae:.6f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    axes[5].axis('off')
    
    plt.tight_layout()
    
    output_path = TRAINING_DIR / 'mae_curves_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ MAE curves saved: {output_path}")
    
    return fig

def plot_threshold_comparison():
    """Plot threshold comparison across time scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Threshold Analysis Across Time Scales', fontsize=16, fontweight='bold')
    
    # Collect data
    global_p95 = []
    global_p99 = []
    global_p99_5 = []
    
    for T in TIME_SCALES:
        thresholds = load_thresholds(T)
        global_p95.append(thresholds['global']['p95'])
        global_p99.append(thresholds['global']['p99'])
        global_p99_5.append(thresholds['global']['p99.5'])
    
    # Plot 1: Global thresholds
    x = np.arange(len(TIME_SCALES))
    width = 0.25
    
    ax1.bar(x - width, global_p95, width, label='p95', color='green', alpha=0.7)
    ax1.bar(x, global_p99, width, label='p99', color='orange', alpha=0.7)
    ax1.bar(x + width, global_p99_5, width, label='p99.5', color='red', alpha=0.7)
    
    ax1.set_xlabel('Time Scale', fontsize=12)
    ax1.set_ylabel('Threshold Value', fontsize=12)
    ax1.set_title('Global Thresholds by Time Scale', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T={t}' for t in TIME_SCALES])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Per-signal thresholds for T=1
    thresholds_t1 = load_thresholds(1)
    
    # Extract signal names and p99 values
    signals = []
    p99_values = []
    
    for signal_name, values in thresholds_t1.items():
        if signal_name not in ['global', 'statistics', 'metadata'] and isinstance(values, dict):
            if 'p99' in values:  # Make sure p99 exists
                signals.append(signal_name)
                p99_values.append(values['p99'])
    
    # Sort by p99 value
    sorted_indices = np.argsort(p99_values)
    signals = [signals[i] for i in sorted_indices]
    p99_values = [p99_values[i] for i in sorted_indices]
    
    # Create color map
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(signals)))
    
    ax2.barh(signals, p99_values, color=colors, alpha=0.7)
    ax2.set_xlabel('p99 Threshold', fontsize=12)
    ax2.set_title('Per-Signal Thresholds (T=1)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(p99_values):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    output_path = TRAINING_DIR / 'threshold_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Threshold analysis saved: {output_path}")
    
    return fig

def plot_final_metrics_comparison():
    """Plot final metrics comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Final Metrics Comparison Across Time Scales', fontsize=16, fontweight='bold')
    
    # Collect data
    final_losses = []
    final_val_losses = []
    final_maes = []
    epochs_trained = []
    
    for T in TIME_SCALES:
        history = load_history(T)
        final_losses.append(history['loss'][-1])
        final_val_losses.append(history['val_loss'][-1])
        final_maes.append(history['mae'][-1])
        epochs_trained.append(len(history['loss']))
    
    x_labels = [f'T={t}' for t in TIME_SCALES]
    
    # Plot 1: Final Loss
    ax1.bar(x_labels, final_losses, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Final Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(final_losses):
        ax1.text(i, v + 0.0002, f'{v:.4f}', ha='center', fontsize=9)
    
    # Plot 2: Final Val Loss
    ax2.bar(x_labels, final_val_losses, color='coral', alpha=0.7)
    ax2.set_ylabel('Validation Loss (MSE)', fontsize=11)
    ax2.set_title('Final Validation Loss', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(final_val_losses):
        ax2.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=9)
    
    # Plot 3: Final MAE
    ax3.bar(x_labels, final_maes, color='mediumseagreen', alpha=0.7)
    ax3.set_ylabel('MAE', fontsize=11)
    ax3.set_title('Final Mean Absolute Error', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(final_maes):
        ax3.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=9)
    
    # Plot 4: Epochs Trained
    ax4.bar(x_labels, epochs_trained, color='mediumpurple', alpha=0.7)
    ax4.set_ylabel('Number of Epochs', fontsize=11)
    ax4.set_title('Epochs Trained (Early Stopping)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(epochs_trained):
        ax4.text(i, v + 2, f'{v}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = TRAINING_DIR / 'final_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Final metrics comparison saved: {output_path}")
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("\n" + "="*80)
    print("PHASE 2: TRAINING RESULTS VISUALIZATION")
    print("="*80)
    print(f"\nReading results from: {TRAINING_DIR}")
    print(f"Time scales: {TIME_SCALES}")
    
    # 1. Print summary
    print("\n[1/5] Generating summary statistics...")
    summary_data = print_training_summary()
    
    # 2. Plot training curves
    print("\n[2/5] Plotting training curves...")
    plot_all_training_curves()
    
    # 3. Plot MAE curves
    print("\n[3/5] Plotting MAE curves...")
    plot_mae_curves()
    
    # 4. Plot thresholds
    print("\n[4/5] Plotting threshold analysis...")
    plot_threshold_comparison()
    
    # 5. Plot final metrics
    print("\n[5/5] Plotting final metrics comparison...")
    plot_final_metrics_comparison()
    
    # Final message
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved in: {TRAINING_DIR}/")
    print("\nGenerated files:")
    print("  - training_curves_all.png")
    print("  - mae_curves_all.png")
    print("  - threshold_analysis.png")
    print("  - final_metrics_comparison.png")
    print("\n" + "="*80)
    print()


if __name__ == '__main__':
    main()
