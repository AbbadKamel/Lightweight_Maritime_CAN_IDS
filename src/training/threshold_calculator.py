"""
Threshold Calculator
====================

Computes reconstruction error thresholds for anomaly detection.
Uses percentile-based approach from CANShield paper.

Threshold Strategy:
    - Per-signal thresholds (individualized for each CAN signal)
    - Global threshold (across all signals)
    - Multiple percentiles (95%, 99%, 99.5%)

Usage in Detection (Phase 3):
    if reconstruction_error > threshold['p99']:
        → INTRUSION DETECTED!
"""

import json
import numpy as np
from pathlib import Path


def compute_reconstruction_errors(autoencoder, X_data, batch_size=128, verbose=0):
    """
    Compute reconstruction errors for input data.
    
    Process:
        1. Forward pass: X_recon = autoencoder.predict(X_data)
        2. Error: |X_data - X_recon| (absolute difference)
        3. Squeeze channel dimension: (N, 50, 15, 1) → (N, 50, 15)
    
    Args:
        autoencoder (keras.Model): Trained autoencoder
        X_data (np.ndarray): Input windows (N, 50, 15, 1)
        batch_size (int): Prediction batch size (default: 128)
        verbose (int): Verbosity (0=silent, 1=progress bar)
    
    Returns:
        np.ndarray: Reconstruction errors (N, 50, 15)
            - N: Number of windows
            - 50: Timesteps
            - 15: Signals
    
    Example:
        >>> errors = compute_reconstruction_errors(model, X_train)
        >>> print(errors.shape)  # (98893, 50, 15)
        >>> print(errors.mean())  # 0.0023
    """
    if verbose > 0:
        print("\nComputing reconstruction errors...")
        print(f"  Input shape: {X_data.shape}")
    
    # Predict reconstructions
    X_recon = autoencoder.predict(X_data, batch_size=batch_size, verbose=verbose)
    
    # Compute absolute errors
    errors = np.abs(X_data - X_recon)
    
    # Remove channel dimension: (N, 50, 15, 1) → (N, 50, 15)
    errors = errors.squeeze(axis=-1)
    
    if verbose > 0:
        print(f"  Output shape: {errors.shape}")
        print(f"  Error range: [{errors.min():.6f}, {errors.max():.6f}]")
        print(f"  Error mean: {errors.mean():.6f}")
        print(f"  Error std: {errors.std():.6f}")
    
    return errors


def calculate_thresholds(errors, signal_names=None, percentiles=[95, 99, 99.5]):
    """
    Calculate percentile-based thresholds.
    
    Strategy:
        - Per-signal thresholds: Computed independently for each signal
        - Global threshold: Computed across all signals
        - Multiple percentiles: 95%, 99%, 99.5%
    
    Args:
        errors (np.ndarray): Reconstruction errors (N, 50, 15)
        signal_names (list): Optional signal names (default: signal_0, signal_1, ...)
        percentiles (list): Percentile values (default: [95, 99, 99.5])
    
    Returns:
        dict: Nested dictionary structure:
            {
                'signal_0': {'p95': 0.023, 'p99': 0.045, 'p99.5': 0.067},
                'signal_1': {'p95': 0.019, 'p99': 0.034, 'p99.5': 0.050},
                ...
                'signal_14': {...},
                'global': {'p95': 0.030, 'p99': 0.051, 'p99.5': 0.072}
            }
    
    Interpretation:
        - p95: Catches 95% of normal behavior (flags 5% as potential anomaly)
        - p99: Stricter threshold (~3σ in normal distribution)
        - p99.5: Very strict (flags only 0.5% of normal data)
    
    Example:
        >>> errors = compute_reconstruction_errors(model, X_train)
        >>> thresholds = calculate_thresholds(errors)
        >>> print(thresholds['global']['p99'])  # 0.0512
    """
    num_windows, time_steps, num_signals = errors.shape
    
    if signal_names is None:
        signal_names = [f'signal_{i}' for i in range(num_signals)]
    
    if len(signal_names) != num_signals:
        raise ValueError(
            f"signal_names length ({len(signal_names)}) must match "
            f"num_signals ({num_signals})"
        )
    
    # Flatten temporal dimension: (N, 50, 15) → (N*50, 15)
    # This pools all timesteps together for threshold calculation
    errors_flat = errors.reshape(-1, num_signals)
    
    thresholds = {}
    
    # ========================================================================
    # PER-SIGNAL THRESHOLDS
    # ========================================================================
    
    for signal_idx, signal_name in enumerate(signal_names):
        signal_errors = errors_flat[:, signal_idx]
        
        thresholds[signal_name] = {
            f'p{p}': float(np.percentile(signal_errors, p))
            for p in percentiles
        }
    
    # ========================================================================
    # GLOBAL THRESHOLD (across all signals)
    # ========================================================================
    
    # Flatten completely: (N*50, 15) → (N*50*15,)
    all_errors = errors_flat.flatten()
    
    thresholds['global'] = {
        f'p{p}': float(np.percentile(all_errors, p))
        for p in percentiles
    }
    
    # ========================================================================
    # ADDITIONAL STATISTICS (for reference)
    # ========================================================================
    
    thresholds['statistics'] = {
        'total_samples': int(num_windows),
        'timesteps': int(time_steps),
        'num_signals': int(num_signals),
        'total_values': int(num_windows * time_steps * num_signals),
        'mean_error': float(all_errors.mean()),
        'std_error': float(all_errors.std()),
        'min_error': float(all_errors.min()),
        'max_error': float(all_errors.max()),
        'percentiles_used': percentiles
    }
    
    return thresholds


def save_thresholds(thresholds, output_dir, time_scale, verbose=1):
    """
    Save thresholds to JSON file.
    
    Args:
        thresholds (dict): Threshold dictionary from calculate_thresholds()
        output_dir (Path|str): Output directory
        time_scale (int): Time scale identifier (1, 5, 10, 20, 50)
        verbose (int): Verbosity
    
    Returns:
        Path: Path to saved JSON file
    
    Example:
        >>> thresholds = calculate_thresholds(errors)
        >>> save_thresholds(thresholds, 'results/training/thresholds', 1)
        ✓ Thresholds saved: results/training/thresholds/thresholds_T1.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    threshold_path = output_dir / f'thresholds_T{time_scale}.json'
    
    # Add metadata
    thresholds['metadata'] = {
        'time_scale': time_scale,
        'file_version': '1.0',
        'description': 'Reconstruction error thresholds for anomaly detection'
    }
    
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    if verbose > 0:
        print(f"✓ Thresholds saved: {threshold_path}")
        
        # Print summary
        if 'global' in thresholds and 'statistics' in thresholds:
            print(f"\n  Global Thresholds:")
            for key, val in thresholds['global'].items():
                print(f"    {key:6s}: {val:.6f}")
            
            print(f"\n  Statistics:")
            stats = thresholds['statistics']
            print(f"    Total samples: {stats['total_samples']:,}")
            print(f"    Mean error:    {stats['mean_error']:.6f}")
            print(f"    Std error:     {stats['std_error']:.6f}")
    
    return threshold_path


def load_thresholds(threshold_path):
    """
    Load thresholds from JSON file.
    
    Args:
        threshold_path (Path|str): Path to threshold JSON file
    
    Returns:
        dict: Threshold dictionary
    
    Example:
        >>> thresholds = load_thresholds('results/training/thresholds/thresholds_T1.json')
        >>> print(thresholds['global']['p99'])
    """
    threshold_path = Path(threshold_path)
    
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")
    
    with open(threshold_path, 'r') as f:
        thresholds = json.load(f)
    
    print(f"✓ Thresholds loaded from {threshold_path}")
    
    return thresholds


def check_anomaly(error, threshold, threshold_type='p99'):
    """
    Check if error exceeds threshold (for detection phase).
    
    Args:
        error (float|np.ndarray): Reconstruction error(s)
        threshold (dict): Threshold dictionary
        threshold_type (str): Which percentile to use (default: 'p99')
    
    Returns:
        bool|np.ndarray: True if anomaly detected
    
    Example:
        >>> thresholds = load_thresholds('thresholds_T1.json')
        >>> is_anomaly = check_anomaly(0.08, thresholds['global'], 'p99')
        >>> print(is_anomaly)  # True (if 0.08 > threshold)
    """
    if threshold_type not in threshold:
        raise ValueError(
            f"Threshold type '{threshold_type}' not found. "
            f"Available: {list(threshold.keys())}"
        )
    
    threshold_value = threshold[threshold_type]
    
    return error > threshold_value


# Quick test function
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from training.autoencoder_builder import build_2d_cnn_autoencoder, compile_autoencoder
    
    print("="*80)
    print("Testing Threshold Calculator")
    print("="*80)
    
    # Create dummy model and data
    print("\nCreating dummy data...")
    np.random.seed(42)
    X_dummy = np.random.rand(1000, 50, 15, 1).astype('float32')
    
    print("Building autoencoder...")
    model = build_2d_cnn_autoencoder(50, 15)
    model = compile_autoencoder(model)
    
    # Compute errors (without training - random predictions)
    print("\nComputing reconstruction errors...")
    errors = compute_reconstruction_errors(model, X_dummy, verbose=1)
    
    # Calculate thresholds
    print("\nCalculating thresholds...")
    signal_names = [
        'wind_speed', 'wind_angle', 'yaw', 'pitch', 'roll',
        'heading', 'variation', 'rate_of_turn', 'cog', 'sog',
        'rudder_angle_order', 'rudder_position', 'latitude', 'longitude', 'depth'
    ]
    
    thresholds = calculate_thresholds(errors, signal_names=signal_names)
    
    print("\nGlobal Thresholds:")
    for key, val in thresholds['global'].items():
        print(f"  {key}: {val:.6f}")
    
    print("\nPer-Signal Thresholds (first 3 signals):")
    for i in range(3):
        sig_name = signal_names[i]
        print(f"  {sig_name}:")
        for key, val in thresholds[sig_name].items():
            print(f"    {key}: {val:.6f}")
    
    # Save and load test
    print("\nTesting save/load...")
    test_dir = Path('test_output/thresholds')
    saved_path = save_thresholds(thresholds, test_dir, time_scale=1, verbose=1)
    
    loaded = load_thresholds(saved_path)
    print(f"\n✓ Loaded thresholds match: {loaded['global'] == thresholds['global']}")
    
    # Anomaly check test
    print("\nTesting anomaly detection...")
    test_error = 0.08
    is_anomaly = check_anomaly(test_error, thresholds['global'], 'p99')
    print(f"  Error: {test_error:.6f}")
    print(f"  Threshold (p99): {thresholds['global']['p99']:.6f}")
    print(f"  Is anomaly? {is_anomaly}")
    
    print("\n✓ Threshold calculator test complete!")
