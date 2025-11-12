#!/usr/bin/env python3
"""
Phase 3 - Step 2: Multi-Scale Detection System
===============================================

This script implements the detection system using trained autoencoders:
1. Load 5 trained autoencoders (T=1,5,10,20,50)
2. Load attack-injected dataset
3. Reconstruct data using each autoencoder
4. Calculate reconstruction errors
5. Compare against thresholds
6. Multi-scale voting (3/5 autoencoders agree)
7. Generate final predictions

Input:  - results/training/models/*.h5 (trained autoencoders)
        - results/training/thresholds/*.json (thresholds)
        - data/attacks/attack_dataset.npz (labeled attack data)

Output: - results/detection/predictions.npz (predictions)
        - results/detection/reconstruction_errors.npz (errors)
        - results/detection/detection_summary.txt (summary)
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'results' / 'training' / 'models'
THRESHOLDS_DIR = BASE_DIR / 'results' / 'training' / 'thresholds'
ATTACKS_DIR = BASE_DIR / 'data' / 'attacks'
DETECTION_DIR = BASE_DIR / 'results' / 'detection'

# Detection parameters
TIME_SCALES = [1, 5, 10, 20, 50]
WINDOW_LENGTH = 50
THRESHOLD_PERCENTILE = 'p99'  # Use p99 threshold (99th percentile)
VOTING_THRESHOLD = 3  # 3 out of 5 autoencoders must agree

# Memory optimization: Process in batches
BATCH_SIZE = 10000  # Process 10K windows at a time
DTYPE = np.float32  # Use float32 instead of float64

# Test size (set to None to use all data, or a number to limit for faster testing)
TEST_SIZE = None  # Use all data for final evaluation

# ============================================================================
# Helper Functions
# ============================================================================

def load_thresholds(threshold_file):
    """Load thresholds from JSON file"""
    with open(threshold_file, 'r') as f:
        thresholds = json.load(f)
    return thresholds


def create_windows(data, window_length):
    """
    Create sliding windows from timestep data (MEMORY EFFICIENT VERSION)
    
    Args:
        data: Array of shape (num_timesteps, num_signals)
        window_length: Length of each window
    
    Returns:
        windows: Array of shape (num_windows, num_signals, window_length, 1)
        window_indices: Start index of each window in original data
    """
    num_timesteps, num_signals = data.shape
    num_windows = num_timesteps - window_length + 1
    
    # Use float32 to save memory
    windows = np.zeros((num_windows, num_signals, window_length, 1), dtype=DTYPE)
    window_indices = np.arange(num_windows)
    
    print(f"    Creating {num_windows:,} windows...")
    print(f"    Memory required: ~{(windows.nbytes / 1024**3):.2f} GB")
    
    for i in range(num_windows):
        if i % 500000 == 0 and i > 0:
            print(f"      Progress: {i:,} / {num_windows:,} ({100*i/num_windows:.1f}%)")
        # Extract window [i:i+window_length]
        window = data[i:i+window_length, :]  # (window_length, num_signals)
        windows[i, :, :, 0] = window.T  # Transpose to (num_signals, window_length)
    
    return windows, window_indices


def reconstruct_with_autoencoder(model, windows):
    """
    Reconstruct windows using autoencoder
    
    Args:
        model: Trained autoencoder
        windows: Input windows (num_windows, num_signals, window_length, 1)
    
    Returns:
        reconstructions: Reconstructed windows (same shape as input)
    """
    reconstructions = model.predict(windows, verbose=0, batch_size=256)
    return reconstructions


def calculate_reconstruction_errors(original, reconstructed):
    """
    Calculate per-timestep reconstruction errors (MAE)
    
    Args:
        original: Original windows (num_windows, num_signals, window_length, 1)
        reconstructed: Reconstructed windows (same shape)
    
    Returns:
        errors: Per-timestep errors (num_windows, window_length)
    """
    # Calculate absolute error for each signal and timestep
    abs_errors = np.abs(original - reconstructed)  # (num_windows, num_signals, window_length, 1)
    
    # Average across signals for each timestep
    timestep_errors = np.mean(abs_errors[:, :, :, 0], axis=1)  # (num_windows, window_length)
    
    return timestep_errors


def apply_threshold(errors, threshold_value):
    """
    Apply threshold to errors to get binary predictions
    
    Args:
        errors: Reconstruction errors (num_windows, window_length)
        threshold_value: Threshold value
    
    Returns:
        predictions: Binary predictions (num_windows, window_length)
                    0 = Normal, 1 = Anomaly
    """
    predictions = (errors > threshold_value).astype(np.int32)
    return predictions


def multi_scale_voting(predictions_dict):
    """
    Combine predictions from multiple time scales using voting
    
    Args:
        predictions_dict: Dictionary of predictions from each time scale
                         {T: predictions array of shape (num_windows, window_length)}
    
    Returns:
        final_predictions: Voted predictions (num_timesteps,)
    """
    # Find the maximum number of timesteps across all scales
    max_timesteps = 0
    for T, preds in predictions_dict.items():
        num_windows, window_length = preds.shape
        timesteps = num_windows + window_length - 1
        max_timesteps = max(max_timesteps, timesteps)
    
    # Create vote matrix (num_scales, num_timesteps)
    num_scales = len(predictions_dict)
    vote_matrix = np.zeros((num_scales, max_timesteps), dtype=np.int32)
    
    # Fill vote matrix
    for scale_idx, (T, preds) in enumerate(predictions_dict.items()):
        num_windows, window_length = preds.shape
        
        # For each timestep, aggregate predictions from all windows containing it
        for t in range(max_timesteps):
            votes = []
            for w in range(num_windows):
                window_start = w
                window_end = w + window_length
                if window_start <= t < window_end:
                    # This window contains timestep t
                    timestep_in_window = t - window_start
                    votes.append(preds[w, timestep_in_window])
            
            if votes:
                # Use majority vote within this scale
                vote_matrix[scale_idx, t] = int(np.mean(votes) >= 0.5)
    
    # Final voting: 3 out of 5 scales must agree
    final_predictions = (np.sum(vote_matrix, axis=0) >= VOTING_THRESHOLD).astype(np.int32)
    
    return final_predictions, vote_matrix


# ============================================================================
# Main Detection Pipeline
# ============================================================================

def run_detection():
    """Main detection pipeline"""
    
    print("\n" + "="*70)
    print("PHASE 3 - STEP 2: MULTI-SCALE DETECTION SYSTEM")
    print("="*70)
    
    # Create output directory
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. Load Attack Dataset
    # ========================================================================
    print("\n[1/6] Loading attack dataset...")
    
    attack_file = ATTACKS_DIR / 'attack_dataset.npz'
    if not attack_file.exists():
        raise FileNotFoundError(f"Attack dataset not found: {attack_file}")
    
    data = np.load(attack_file, allow_pickle=True)
    X_data = data['X_data']
    y_labels = data['y_labels']
    signal_names = data['signal_names']
    label_map = data['label_map'].item()
    
    # Limit dataset size for testing if specified
    if TEST_SIZE is not None and TEST_SIZE < len(X_data):
        print(f"  NOTE: Using subset of {TEST_SIZE:,} samples for testing")
        indices = np.random.choice(len(X_data), size=TEST_SIZE, replace=False)
        X_data = X_data[indices]
        y_labels = y_labels[indices]
    
    num_timesteps, num_signals = X_data.shape
    
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Number of signals: {num_signals}")
    print(f"  Label distribution:")
    for label_name, label_value in sorted(label_map.items(), key=lambda x: x[1]):
        count = np.sum(y_labels == label_value)
        percentage = (count / len(y_labels)) * 100
        print(f"    {label_name.capitalize():12s} (label={label_value}): {count:8,} ({percentage:5.2f}%)")
    
    # ========================================================================
    # 2. Load Models and Thresholds
    # ========================================================================
    print("\n[2/6] Loading trained models and thresholds...")
    
    models = {}
    thresholds = {}
    
    for T in TIME_SCALES:
        # Load model
        model_file = MODELS_DIR / f'autoencoder_T{T}_best.h5'
        if not model_file.exists():
            print(f"  WARNING: Model not found: {model_file}, skipping T={T}")
            continue
        
        models[T] = keras.models.load_model(model_file, compile=False)
        print(f"  Loaded model T={T:2d}: {model_file.name}")
        
        # Load thresholds
        threshold_file = THRESHOLDS_DIR / f'thresholds_T{T}.json'
        if not threshold_file.exists():
            print(f"  WARNING: Thresholds not found: {threshold_file}")
            continue
        
        threshold_data = load_thresholds(threshold_file)
        # Extract global threshold for the selected percentile
        if 'global' in threshold_data and THRESHOLD_PERCENTILE in threshold_data['global']:
            thresholds[T] = threshold_data['global'][THRESHOLD_PERCENTILE]
            print(f"    Threshold ({THRESHOLD_PERCENTILE}): {thresholds[T]:.6f}")
        else:
            print(f"  WARNING: Could not find {THRESHOLD_PERCENTILE} threshold for T={T}")
    
    if not models:
        raise ValueError("No models loaded! Check model files.")
    
    print(f"\n  Successfully loaded {len(models)} models")
    
    # ========================================================================
    # 3. Create Windows for Detection
    # ========================================================================
    print("\n[3/6] Creating sliding windows...")
    
    windows, window_indices = create_windows(X_data, WINDOW_LENGTH)
    num_windows = len(windows)
    
    print(f"  Input shape: {X_data.shape} (timesteps)")
    print(f"  Windows shape: {windows.shape}")
    print(f"  Number of windows: {num_windows:,}")
    
    # ========================================================================
    # 4. Run Detection for Each Time Scale
    # ========================================================================
    print("\n[4/6] Running detection for each time scale...")
    
    predictions_dict = {}
    errors_dict = {}
    
    for T in sorted(models.keys()):
        print(f"\n  [Time Scale T={T}]")
        
        # Reconstruct
        print(f"    Reconstructing {num_windows:,} windows...")
        reconstructions = reconstruct_with_autoencoder(models[T], windows)
        
        # Calculate errors
        print(f"    Calculating reconstruction errors...")
        errors = calculate_reconstruction_errors(windows, reconstructions)
        errors_dict[T] = errors
        
        # Apply threshold
        print(f"    Applying threshold ({THRESHOLD_PERCENTILE} = {thresholds[T]:.6f})...")
        predictions = apply_threshold(errors, thresholds[T])
        predictions_dict[T] = predictions
        
        # Statistics
        num_anomalies = np.sum(predictions)
        total_predictions = predictions.size
        anomaly_rate = (num_anomalies / total_predictions) * 100
        
        print(f"    Detected anomalies: {num_anomalies:,} / {total_predictions:,} ({anomaly_rate:.2f}%)")
    
    # ========================================================================
    # 5. Multi-Scale Voting
    # ========================================================================
    print("\n[5/6] Performing multi-scale voting...")
    
    final_predictions, vote_matrix = multi_scale_voting(predictions_dict)
    
    # Trim predictions to match original data length
    final_predictions = final_predictions[:num_timesteps]
    vote_matrix = vote_matrix[:, :num_timesteps]
    
    print(f"  Voting threshold: {VOTING_THRESHOLD}/{len(models)} scales must agree")
    print(f"  Final predictions shape: {final_predictions.shape}")
    
    # Statistics
    num_detected_anomalies = np.sum(final_predictions == 1)
    detection_rate = (num_detected_anomalies / num_timesteps) * 100
    
    print(f"\n  Detection Summary:")
    print(f"    Total timesteps: {num_timesteps:,}")
    print(f"    Detected as Normal: {np.sum(final_predictions == 0):,} ({100 - detection_rate:.2f}%)")
    print(f"    Detected as Anomaly: {num_detected_anomalies:,} ({detection_rate:.2f}%)")
    
    # ========================================================================
    # 6. Save Results
    # ========================================================================
    print("\n[6/6] Saving detection results...")
    
    # Save predictions
    predictions_file = DETECTION_DIR / 'predictions.npz'
    np.savez_compressed(
        predictions_file,
        y_true=y_labels,
        y_pred=final_predictions,
        vote_matrix=vote_matrix,
        signal_names=signal_names,
        label_map=label_map,
        time_scales=TIME_SCALES
    )
    print(f"  Predictions saved: {predictions_file}")
    
    # Save reconstruction errors
    errors_file = DETECTION_DIR / 'reconstruction_errors.npz'
    np.savez_compressed(errors_file, **errors_dict)
    print(f"  Reconstruction errors saved: {errors_file}")
    
    # Save detection summary
    summary_file = DETECTION_DIR / 'detection_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DETECTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Detection Date: {datetime.now().isoformat()}\n")
        f.write(f"Total Timesteps: {num_timesteps:,}\n")
        f.write(f"Number of Signals: {num_signals}\n")
        f.write(f"Window Length: {WINDOW_LENGTH}\n")
        f.write(f"Time Scales: {TIME_SCALES}\n")
        f.write(f"Threshold Used: {THRESHOLD_PERCENTILE}\n")
        f.write(f"Voting Threshold: {VOTING_THRESHOLD}/{len(models)}\n\n")
        
        f.write("Ground Truth Distribution:\n")
        for label_name, label_value in sorted(label_map.items(), key=lambda x: x[1]):
            count = np.sum(y_labels == label_value)
            percentage = (count / len(y_labels)) * 100
            f.write(f"  {label_name.capitalize():12s} (label={label_value}): {count:8,} ({percentage:5.2f}%)\n")
        
        f.write(f"\nDetection Results:\n")
        f.write(f"  Detected as Normal:  {np.sum(final_predictions == 0):,} ({100 - detection_rate:.2f}%)\n")
        f.write(f"  Detected as Anomaly: {num_detected_anomalies:,} ({detection_rate:.2f}%)\n\n")
        
        f.write("Per-Scale Detection Rates:\n")
        for T in sorted(models.keys()):
            preds = predictions_dict[T]
            anomaly_count = np.sum(preds)
            rate = (anomaly_count / preds.size) * 100
            f.write(f"  T={T:2d}: {anomaly_count:8,} anomalies ({rate:5.2f}%)\n")
    
    print(f"  Summary saved: {summary_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("DETECTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Detection Summary:")
    print(f"  Models used: {len(models)} (T={sorted(models.keys())})")
    print(f"  Timesteps processed: {num_timesteps:,}")
    print(f"  Windows created: {num_windows:,}")
    print(f"\n🎯 Detection Results:")
    print(f"  Normal:  {np.sum(final_predictions == 0):,} ({100 - detection_rate:.2f}%)")
    print(f"  Anomaly: {num_detected_anomalies:,} ({detection_rate:.2f}%)")
    print(f"\n💾 Output Files:")
    print(f"  Predictions: {predictions_file}")
    print(f"  Errors: {errors_file}")
    print(f"  Summary: {summary_file}")
    print(f"\n✅ Ready for Phase 3 - Step 3: Evaluation Metrics")
    print("="*70 + "\n")
    
    return final_predictions, y_labels


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    import os
    
    try:
        run_detection()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
