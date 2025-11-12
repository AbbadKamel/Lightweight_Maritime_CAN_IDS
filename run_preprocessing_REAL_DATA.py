"""
PHASE 1 PREPROCESSING - REAL MARITIME DATA
==========================================

This script processes YOUR REAL decoded CAN data from:
  - decoded_brute_frames.csv (98,943 messages)
  - Uses signal_order.txt from initialization

Outputs:
  1. Preprocessed training windows (.npy files)
  2. Normalization parameters (min/max per signal)
  3. Comprehensive visualizations of REAL data
  4. Quality report

Author: CANShield Pipeline
Date: 2025-11-10
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'CANShield' / 'src'))

from preprocessing.data_loader import CANDataLoader, load_signal_order
from preprocessing.visualize import (
    plot_signal_timeseries,
    plot_normalization_effect,
    plot_multiscale_comparison,
    plot_distribution_comparison,
    plot_window_as_heatmap,
    plot_multiple_windows
)


def main():
    """Process REAL maritime CAN data through Phase 1 preprocessing"""
    
    print("="*80)
    print("PHASE 1 PREPROCESSING - REAL MARITIME DATA")
    print("="*80)
    print()
    
    # ==========================================================================
    # STEP 1: Define paths to YOUR REAL data
    # ==========================================================================
    base_dir = Path(__file__).parent
    
    # REAL decoded CSV file
    csv_path = base_dir / 'results' / 'fixed_decoder_data' / 'decoded_brute_frames.csv'
    
    # Signal order from initialization
    signal_order_path = base_dir / 'results' / 'initialization' / 'signal_order.txt'
    
    # Output directories
    output_dir = base_dir / 'results' / 'preprocessing'
    vis_dir = output_dir / 'visualizations'
    data_dir = output_dir / 'windows'
    params_dir = output_dir / 'parameters'
    
    # Create directories
    for d in [output_dir, vis_dir, data_dir, params_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Input CSV:     {csv_path}")
    print(f"✓ Signal order:  {signal_order_path}")
    print(f"✓ Output dir:    {output_dir}")
    print()
    
    # Verify files exist
    if not csv_path.exists():
        print(f"❌ ERROR: Decoded CSV not found at {csv_path}")
        return
    
    if not signal_order_path.exists():
        print(f"❌ ERROR: Signal order file not found at {signal_order_path}")
        return
    
    # ==========================================================================
    # STEP 2: Load signal order (15 signals)
    # ==========================================================================
    print("="*80)
    print("STEP 1: Loading signal order...")
    print("="*80)
    
    signal_names = load_signal_order(signal_order_path)
    print(f"✓ Loaded {len(signal_names)} signals:")
    for i, name in enumerate(signal_names, 1):
        print(f"  {i:2d}. {name}")
    print()
    
    # ==========================================================================
    # STEP 3: Load REAL decoded data
    # ==========================================================================
    print("="*80)
    print("STEP 2: Loading REAL maritime CAN data...")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"✓ Time range: {df['Timestamp'].iloc[0]} → {df['Timestamp'].iloc[-1]}")
    print()
    
    # Check data coverage
    print("Signal coverage (non-null %):")
    for signal in signal_names:
        if signal in df.columns:
            coverage = (df[signal].notna().sum() / len(df)) * 100
            print(f"  {signal:20s}: {coverage:6.2f}%")
        else:
            print(f"  {signal:20s}: NOT FOUND ❌")
    print()
    
    # ==========================================================================
    # STEP 4: Initialize data loader
    # ==========================================================================
    print("="*80)
    print("STEP 3: Initializing preprocessing pipeline...")
    print("="*80)
    
    loader = CANDataLoader(
        signal_names=signal_names,
        window_size=50,
        sampling_periods=[1, 5, 10, 20, 50]
    )
    
    print(f"✓ Window size:      {loader.window_size}")
    print(f"✓ Sampling periods: {loader.sampling_periods}")
    print(f"✓ Number of signals: {loader.num_signals}")
    print()
    
    # ==========================================================================
    # STEP 5: Preprocess data (multi-scale views + forward-fill)
    # ==========================================================================
    print("="*80)
    print("STEP 4: Preprocessing REAL data...")
    print("="*80)
    print("This may take a few minutes for 98,943 messages...")
    print()
    
    # Load and preprocess (training mode: apply backward fill)
    windows_dict = loader.load_and_preprocess(
        csv_path=str(csv_path),
        apply_bfill=True  # Training mode: can use future data in CSV
    )
    
    print("\n✓ Preprocessing complete!")
    print(f"✓ Generated windows for {len(windows_dict)} sampling periods:")
    for period, windows in windows_dict.items():
        print(f"  T={period:2d}: {windows.shape[0]:6d} windows × {windows.shape[1]} signals × {windows.shape[2]} timesteps")
    print()
    
    # ==========================================================================
    # STEP 6: Fit normalizers on REAL data
    # ==========================================================================
    print("="*80)
    print("STEP 5: Fitting normalization parameters...")
    print("="*80)
    
    loader.fit_normalizers(windows_dict)
    normalizers = loader.normalizers
    
    print(f"✓ Fitted {len(normalizers)} normalizers")
    print("\nMin/Max values from REAL data:")
    for period, norm in normalizers.items():
        print(f"\n  T={period}:")
        for signal in signal_names:
            min_val = norm.min_values[signal]
            max_val = norm.max_values[signal]
            print(f"    {signal:20s}: [{min_val:10.4f}, {max_val:10.4f}]")
    print()
    
    # ==========================================================================
    # STEP 7: Normalize windows
    # ==========================================================================
    print("="*80)
    print("STEP 6: Normalizing windows to [0,1]...")
    print("="*80)
    
    normalized_dict = loader.transform_windows(windows_dict)
    
    print("✓ Normalization complete!")
    for period, windows in normalized_dict.items():
        min_val = windows.min()
        max_val = windows.max()
        mean_val = windows.mean()
        print(f"  T={period:2d}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    print()
    
    # ==========================================================================
    # STEP 8: Save windows and parameters
    # ==========================================================================
    print("="*80)
    print("STEP 7: Saving preprocessed data...")
    print("="*80)
    
    for period in loader.sampling_periods:
        # Save normalized windows
        windows_file = data_dir / f'windows_T{period}_normalized.npy'
        np.save(windows_file, normalized_dict[period])
        print(f"✓ Saved {windows_file.name} ({normalized_dict[period].shape})")
        
        # Save normalization parameters
        params_file = params_dir / f'normalization_params_T{period}.csv'
        normalizers[period].save_parameters(str(params_file))
        print(f"  + Parameters: {params_file.name}")
    print()
    
    # ==========================================================================
    # STEP 9: Generate visualizations of REAL data
    # ==========================================================================
    print("="*80)
    print("STEP 8: Generating visualizations of REAL maritime data...")
    print("="*80)
    print()
    
    # Use T=1 (highest resolution) for visualizations
    raw_windows = windows_dict[1]
    norm_windows = normalized_dict[1]
    
    # -------------------------------------------------------------------------
    # 9.1: Time series of REAL signals
    # -------------------------------------------------------------------------
    print("Generating time series plots...")
    
    # Take first 5000 timesteps from first window
    if len(raw_windows) > 0:
        sample_data = raw_windows[0]  # Shape: (15, 50)
        
        fig = plot_signal_timeseries(
            data=sample_data,
            signal_names=signal_names,
            title="REAL Maritime CAN Signals - First Window (T=1)",
            figsize=(15, 12)
        )
        plt.savefig(vis_dir / '01_real_signals_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 01_real_signals_timeseries.png")
    
    # -------------------------------------------------------------------------
    # 9.2: Normalization effect
    # -------------------------------------------------------------------------
    print("Generating normalization comparison...")
    
    if len(raw_windows) >= 10:
        # Take multiple windows for better visualization
        # Reshape to (num_signals, total_timesteps)
        sample_raw = raw_windows[:10].reshape(15, -1)  # 10 windows × 50 timesteps = 500 timesteps
        sample_norm = norm_windows[:10].reshape(15, -1)
        
        fig = plot_normalization_effect(
            original=sample_raw,
            normalized=sample_norm,
            signal_names=signal_names
        )
        plt.savefig(vis_dir / '02_normalization_effect.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 02_normalization_effect.png")
    
    # -------------------------------------------------------------------------
    # 9.3: Multi-scale comparison
    # -------------------------------------------------------------------------
    print("Generating multi-scale comparison...")
    
    # Create multi-scale views dict for first window
    if all(len(windows_dict[T]) > 0 for T in loader.sampling_periods):
        multi_scale_views = {}
        for period in loader.sampling_periods:
            # Get first window: shape (15, 50)
            multi_scale_views[period] = windows_dict[period][0]
        
        # Plot for first 3 signals
        for i in range(min(3, len(signal_names))):
            fig = plot_multiscale_comparison(
                views=multi_scale_views,
                signal_names=signal_names,
                signal_index=i
            )
            signal_name = signal_names[i]
            plt.savefig(vis_dir / f'03_multiscale_{signal_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: 03_multiscale_{signal_name}.png")
    
    # -------------------------------------------------------------------------
    # 9.4: Distribution comparison
    # -------------------------------------------------------------------------
    print("Generating distribution plots...")
    
    if len(raw_windows) > 0:
        # Create distribution data dict
        dist_data = {
            'Raw (T=1)': raw_windows,
            'Normalized (T=1)': norm_windows
        }
        
        # Plot for first 3 signals
        for i in range(min(3, len(signal_names))):
            fig = plot_distribution_comparison(
                data_dict=dist_data,
                signal_names=signal_names,
                signal_index=i
            )
            signal_name = signal_names[i]
            plt.savefig(vis_dir / f'04_distribution_{signal_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: 04_distribution_{signal_name}.png")
    
    # -------------------------------------------------------------------------
    # 9.5: Sample windows as heatmaps
    # -------------------------------------------------------------------------
    print("Generating window heatmaps...")
    
    if len(norm_windows) >= 4:
        fig = plot_multiple_windows(
            windows=norm_windows,
            signal_names=signal_names,
            num_windows=4
        )
        plt.savefig(vis_dir / '05_sample_windows_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 05_sample_windows_heatmap.png")
    
    # Single window detailed view
    if len(norm_windows) > 0:
        fig = plot_window_as_heatmap(
            window=norm_windows[0],
            signal_names=signal_names
        )
        plt.savefig(vis_dir / '06_single_window_detail.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 06_single_window_detail.png")
    
    print()
    
    # ==========================================================================
    # STEP 10: Generate summary report
    # ==========================================================================
    print("="*80)
    print("STEP 9: Generating summary report...")
    print("="*80)
    
    report = []
    report.append("="*80)
    report.append("PHASE 1 PREPROCESSING - SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    report.append("INPUT DATA:")
    report.append(f"  Source:         {csv_path.name}")
    report.append(f"  Total messages: {len(df):,}")
    report.append(f"  Signals:        {len(signal_names)}")
    report.append(f"  Time range:     {df['Timestamp'].iloc[0]} → {df['Timestamp'].iloc[-1]}")
    report.append("")
    
    report.append("PREPROCESSING CONFIGURATION:")
    report.append(f"  Window size:      {loader.window_size}")
    report.append(f"  Sampling periods: {loader.sampling_periods}")
    report.append(f"  Number of signals: {loader.num_signals}")
    report.append(f"  Forward-fill:     Enabled")
    report.append(f"  Backward-fill:    Enabled (training mode)")
    report.append("")
    
    report.append("OUTPUT WINDOWS:")
    for period in loader.sampling_periods:
        shape = normalized_dict[period].shape
        report.append(f"  T={period:2d}: {shape[0]:6d} windows × {shape[1]:2d} signals × {shape[2]:2d} timesteps")
    report.append("")
    
    report.append("NORMALIZATION RANGES (from REAL data):")
    for signal in signal_names:
        min_val = normalizers[1].min_values[signal]
        max_val = normalizers[1].max_values[signal]
        report.append(f"  {signal:20s}: [{min_val:10.4f}, {max_val:10.4f}]")
    report.append("")
    
    report.append("SAVED FILES:")
    report.append(f"  Windows:     {data_dir} (5 files)")
    report.append(f"  Parameters:  {params_dir} (5 files)")
    report.append(f"  Plots:       {vis_dir} (12+ files)")
    report.append("")
    
    report.append("NEXT PHASE:")
    report.append("  PHASE 2: CNN Autoencoder Training")
    report.append("    - Train AE_1, AE_5, AE_10, AE_20, AE_50")
    report.append("    - Compute three-tier thresholds")
    report.append("    - Save trained models")
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    report_file = output_dir / 'preprocessing_summary.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Report saved: {report_file}")
    print()
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("="*80)
    print("✅ PHASE 1 PREPROCESSING COMPLETE!")
    print("="*80)
    print()
    print("REAL maritime data has been successfully preprocessed!")
    print()
    print(f"📊 Generated {sum(w.shape[0] for w in normalized_dict.values()):,} total training windows")
    print(f"📁 Saved to: {output_dir}")
    print(f"🖼️  Visualizations: {vis_dir}")
    print()
    print("Ready for PHASE 2: CNN Autoencoder Training")
    print()
    print("="*80)


if __name__ == '__main__':
    main()
