#!/usr/bin/env python3
"""
Phase 3 - Step 2: Detection System (MEMORY-EFFICIENT VERSION)
==============================================================

This script runs detection on the attack dataset using trained autoencoders.
Uses BATCH PROCESSING to avoid memory issues on low-RAM computers.

Input:  
    - data/attacks/attack_dataset.npz (from Step 1)
    - results/training/models/autoencoder_T*.h5 (from Phase 2)
    - results/training/thresholds/thresholds_T*.json (from Phase 2)

Output:
    - results/detection/predictions.npz (predictions + labels)
    - results/detection/detection_summary.json (metrics)
"""

import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

# Input paths
ATTACKS_DIR = DATA_DIR / 'attacks'
MODELS_DIR = RESULTS_DIR / 'training' / 'models'
THRESHOLDS_DIR = RESULTS_DIR / 'training' / 'thresholds'

# Output paths
DETECTION_DIR = RESULTS_DIR / 'detection'
DETECTION_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
TIME_SCALES = [1, 5, 10, 20, 50]
WINDOW_LENGTH = 50
VOTING_THRESHOLD = 3  # 3 out of 5 autoencoders must agree
THRESHOLD_PERCENTILE = 'p95'  # p95 - lower threshold for detection (accept higher false positives)

# **MEMORY EFFICIENT SETTINGS**
BATCH_SIZE = 1000           # Process 1000 windows at a time
SAVE_EVERY = 10000          # Save progress every 10k windows
MAX_SAMPLES = 50000         # TESTING MODE: Use only 50k samples (set to None for full dataset)

# ============================================================================
# Helper Functions
# ============================================================================

def load_thresholds():
    """Load detection thresholds for all time scales"""
    thresholds = {}
    for T in TIME_SCALES:
        threshold_file = f"results/training/thresholds/thresholds_T{T}.json"
        with open(threshold_file, 'r') as f:
            data = json.load(f)
            
            # Check if THRESHOLD_PERCENTILE exists
            if THRESHOLD_PERCENTILE in data['global']:
                thresholds[f'T{T}'] = data['global'][THRESHOLD_PERCENTILE]
            elif THRESHOLD_PERCENTILE in ['p96', 'p97', 'p98']:
                # Interpolate between p95 and p99
                p95 = data['global']['p95']
                p99 = data['global']['p99']
                percentile_num = int(THRESHOLD_PERCENTILE[1:])
                ratio = (percentile_num - 95) / (99 - 95)  # 0.0 to 1.0
                thresholds[f'T{T}'] = p95 + (p99 - p95) * ratio
            else:
                raise ValueError(f"Threshold {THRESHOLD_PERCENTILE} not found and cannot be interpolated")
    
    return thresholds


def create_windows_from_timesteps(timesteps, window_length=50):
    """
    Convert flat timesteps back to sliding windows
    
    Args:
        timesteps: Array of shape [num_timesteps, num_signals]
        window_length: Window size (default 50)
    
    Returns:
        windows: Array of shape [num_windows, window_length, num_signals]
    """
    num_timesteps, num_signals = timesteps.shape
    num_windows = num_timesteps - window_length + 1
    
    if num_windows <= 0:
        raise ValueError(f"Not enough timesteps ({num_timesteps}) to create windows of length {window_length}")
    
    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(
        timesteps, 
        window_shape=(window_length,), 
        axis=0
    )
    
    # Reshape to [num_windows, window_length, num_signals]
    windows = windows.transpose(0, 2, 1)
    
    return windows.astype(np.float32)


def detect_batch(windows_batch, models, thresholds):
    """
    Detect anomalies in a batch of windows using voting across time scales.
    
    Args:
        windows_batch: Array of shape (N, 50, num_signals) - batch of windows
        models: Dict of {T: autoencoder_model}
        thresholds: Dict of {T: threshold_value}
    
    Returns:
        predictions: Binary array (0=normal, 1=attack)
    """
    N = len(windows_batch)
    votes = np.zeros(N, dtype=np.int32)
    
    # Each autoencoder votes
    for T in TIME_SCALES:
        model = models[f'T{T}']
        threshold = float(thresholds[f'T{T}'])  # Ensure scalar
        
        # Add channel dimension for model input: (N, 50, 15) -> (N, 50, 15, 1)
        batch_input = windows_batch[..., np.newaxis]
        
        # Reconstruct and calculate MSE
        reconstructed = model.predict(batch_input, verbose=0)
        
        # Calculate MSE per window (average over timesteps, signals, and channel)
        mse = np.mean(np.square(batch_input - reconstructed), axis=(1, 2, 3))
        
        # Vote: anomaly if MSE > threshold
        is_anomaly = (mse > threshold).astype(np.int32)
        votes += is_anomaly
    
    # Final decision: attack if >= VOTING_THRESHOLD autoencoders agree
    predictions = (votes >= VOTING_THRESHOLD).astype(np.int32)
    
    return predictions


# ============================================================================
# Main Detection Pipeline
# ============================================================================

def run_detection():
    """Main detection pipeline with memory-efficient batch processing"""
    
    print("\n" + "="*70)
    print("PHASE 3 - STEP 2: DETECTION SYSTEM (MEMORY-EFFICIENT)")
    print("="*70)
    
    # ========================================================================
    # 1. Load Attack Dataset Metadata (NOT the full data yet!)
    # ========================================================================
    print("\n[1/6] Loading dataset metadata...")
    
    attack_file = ATTACKS_DIR / 'attack_dataset.npz'
    if not attack_file.exists():
        raise FileNotFoundError(f"Attack dataset not found: {attack_file}")
    
    # Load only to get shape, then close
    with np.load(attack_file) as data:
        total_timesteps = len(data['y_labels'])
        num_signals = data['X_data'].shape[1]
    
    num_windows = total_timesteps - WINDOW_LENGTH + 1
    
    print(f"  Dataset file: {attack_file}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Number of signals: {num_signals}")
    print(f"  Windows to process: {num_windows:,}")
    
    # Limit samples if specified
    if MAX_SAMPLES is not None and MAX_SAMPLES < num_windows:
        num_windows = MAX_SAMPLES
        print(f"  ⚠️  LIMITED to first {num_windows:,} windows for testing")
    
    # ========================================================================
    # 2. Load Models
    # ========================================================================
    print("\n[2/6] Loading trained models...")
    
    from tensorflow import keras
    
    models = {}
    for T in TIME_SCALES:
        model_file = MODELS_DIR / f'autoencoder_T{T}_best.h5'
        if not model_file.exists():
            model_file = MODELS_DIR / f'autoencoder_T{T}.h5'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        models[f'T{T}'] = keras.models.load_model(model_file, compile=False)
        print(f"  ✓ Loaded AE_T{T}")
    
    # ========================================================================
    # 3. Load Thresholds
    # ========================================================================
    print("\n[3/6] Loading detection thresholds...")
    
    thresholds = load_thresholds()
    for T in TIME_SCALES:
        print(f"  ✓ T={T:2d}: threshold ({THRESHOLD_PERCENTILE}) = {thresholds[f'T{T}']:.6f}")
    
    # ========================================================================
    # 4. Process Data in Batches
    # ========================================================================
    print("\n[4/6] Running detection (batch processing)...")
    print(f"  Batch size: {BATCH_SIZE} windows")
    print(f"  Voting threshold: {VOTING_THRESHOLD}/5 autoencoders")
    print(f"  Saving progress every: {SAVE_EVERY} windows")
    print()
    
    # Load full dataset NOW (we need it)
    print("  Loading attack dataset into memory...")
    data = np.load(attack_file)
    X_timesteps = data['X_data'].astype(np.float32)  # Use float32 to save memory
    y_labels = data['y_labels']
    
    print(f"  ✓ Loaded {len(X_timesteps):,} timesteps")
    
    # Limit if needed
    if MAX_SAMPLES is not None:
        max_timesteps = MAX_SAMPLES + WINDOW_LENGTH - 1
        X_timesteps = X_timesteps[:max_timesteps]
        y_labels = y_labels[:max_timesteps]
        print(f"  ⚠️  Limited to {len(X_timesteps):,} timesteps")
    
    # Create windows from timesteps
    print("  Creating sliding windows...")
    windows = create_windows_from_timesteps(X_timesteps, WINDOW_LENGTH)
    print(f"  ✓ Created {len(windows):,} windows")
    
    # Labels: use the LAST timestep label of each window
    window_labels = y_labels[WINDOW_LENGTH-1:][:len(windows)]
    
    # Arrays to store results
    all_predictions = np.zeros(len(windows), dtype=np.int32)
    
    # Process in batches
    num_batches = (len(windows) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"  DEBUG: windows shape = {windows.shape}")
    print(f"  DEBUG: Processing {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(windows))
        
        batch_windows = windows[start_idx:end_idx]
        
        if batch_idx == 0:
            print(f"  DEBUG: First batch_windows shape = {batch_windows.shape}")
        
        # Run detection on batch
        batch_predictions = detect_batch(batch_windows, models, thresholds)
        
        if batch_idx == 0:
            print(f"  DEBUG: First batch_predictions shape = {batch_predictions.shape}")
        
        all_predictions[start_idx:end_idx] = batch_predictions
        
        # Progress update
        processed = end_idx
        percent = (processed / len(windows)) * 100
        print(f"  Processed: {processed:7,} / {len(windows):,} windows ({percent:5.1f}%)", end='\r')
        
        # Save progress periodically
        if processed % SAVE_EVERY == 0 or processed == len(windows):
            progress_file = DETECTION_DIR / 'predictions_progress.npz'
            np.savez_compressed(
                progress_file,
                predictions=all_predictions[:processed],
                labels=window_labels[:processed],
                processed=processed,
                total=len(windows)
            )
    
    print()  # New line after progress
    print(f"  ✓ Detection complete!")
    
    # ========================================================================
    # 5. Calculate Metrics
    # ========================================================================
    print("\n[5/6] Calculating detection metrics...")
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Binary classification: Normal (0) vs Anomaly (1)
    # Convert multi-class labels to binary (0=Normal, 1-5=Anomaly)
    binary_labels = (window_labels > 0).astype(np.int32)
    
    accuracy = accuracy_score(binary_labels, all_predictions)
    precision = precision_score(binary_labels, all_predictions, zero_division=0)
    recall = recall_score(binary_labels, all_predictions, zero_division=0)
    f1 = f1_score(binary_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(binary_labels, all_predictions)
    
    print(f"\n  Overall Detection Performance:")
    print(f"  {'='*50}")
    print(f"  Accuracy:  {accuracy*100:6.2f}%")
    print(f"  Precision: {precision*100:6.2f}%")
    print(f"  Recall:    {recall*100:6.2f}%")
    print(f"  F1-Score:  {f1*100:6.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:7,}  FP: {cm[0,1]:7,}")
    print(f"    FN: {cm[1,0]:7,}  TP: {cm[1,1]:7,}")
    
    # Per-attack-type breakdown
    print(f"\n  Per-Attack-Type Performance:")
    print(f"  {'='*50}")
    
    label_names = {0: 'Normal', 1: 'Flooding', 2: 'Suppress', 3: 'Plateau', 4: 'Continuous', 5: 'Playback'}
    
    for label_val in range(6):
        mask = (window_labels == label_val)
        if mask.sum() == 0:
            continue
        
        true_positives = ((window_labels == label_val) & (all_predictions == 1)).sum()
        detected_rate = (true_positives / mask.sum()) * 100 if mask.sum() > 0 else 0
        
        print(f"  {label_names[label_val]:12s}: {mask.sum():7,} samples, {detected_rate:5.1f}% detected")
    
    # ========================================================================
    # 6. Save Results
    # ========================================================================
    print(f"\n[6/6] Saving results...")
    
    # Save predictions
    predictions_file = DETECTION_DIR / 'predictions.npz'
    np.savez_compressed(
        predictions_file,
        predictions=all_predictions,
        labels=window_labels,
        binary_labels=binary_labels
    )
    print(f"  ✓ Predictions saved: {predictions_file}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_windows': int(len(windows)),
        'num_signals': int(num_signals),
        'voting_threshold': VOTING_THRESHOLD,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'confusion_matrix': cm.tolist(),
        'per_attack_stats': {}
    }
    
    for label_val in range(6):
        mask = (window_labels == label_val)
        if mask.sum() > 0:
            tp = ((window_labels == label_val) & (all_predictions == 1)).sum()
            summary['per_attack_stats'][label_names[label_val]] = {
                'total': int(mask.sum()),
                'detected': int(tp),
                'rate': float(tp / mask.sum())
            }
    
    summary_file = DETECTION_DIR / 'detection_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved: {summary_file}")
    
    # Clean up progress file
    progress_file = DETECTION_DIR / 'predictions_progress.npz'
    if progress_file.exists():
        progress_file.unlink()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("DETECTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Results Summary:")
    print(f"  Windows processed: {len(windows):,}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"\n💾 Output Files:")
    print(f"  Predictions: {predictions_file}")
    print(f"  Summary:     {summary_file}")
    print(f"\n✅ Ready for Phase 3 - Step 3: Results Visualization")
    print("="*70 + "\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    try:
        run_detection()
    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user")
        print("Progress saved in: results/detection/predictions_progress.npz")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
