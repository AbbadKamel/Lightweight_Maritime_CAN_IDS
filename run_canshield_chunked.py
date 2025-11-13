#!/usr/bin/env python3
"""
CANShield-Style Detection - Memory-Efficient Implementation
============================================================

Uses chunked processing like CANShield to avoid memory issues.
Tests on subset first, then scales to full dataset.

Author: Adapted from CANShield methodology
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Parse command line arguments
if len(sys.argv) != 6:
    print("Usage: python run_canshield_chunked.py <test_size> <loss_factor> <time_factor> <signal_factor> <output_suffix>")
    print()
    print("Examples:")
    print("  Quick test:  python run_canshield_chunked.py 100000 95 95 95 test_100k")
    print("  Validation:  python run_canshield_chunked.py 500000 95 95 95 val_500k")
    print("  Full run:    python run_canshield_chunked.py -1 95 95 95 full")
    print()
    print("Arguments:")
    print("  test_size:     Number of timesteps (-1 for ALL)")
    print("  loss_factor:   Percentile for signal-level threshold (95, 99, etc.)")
    print("  time_factor:   Percentile for temporal threshold (95, 99, etc.)")
    print("  signal_factor: Percentile for window threshold (95, 99, etc.)")
    print("  output_suffix: Name for output files")
    sys.exit(1)

TEST_SIZE = int(sys.argv[1])  # -1 means use all data
LOSS_FACTOR = float(sys.argv[2])
TIME_FACTOR = float(sys.argv[3])
SIGNAL_FACTOR = float(sys.argv[4])
OUTPUT_SUFFIX = sys.argv[5]

print("="*80)
print(f"CANSHIELD CHUNKED DETECTION: {OUTPUT_SUFFIX}")
print("="*80)
print(f"  Test size:          {TEST_SIZE:,} timesteps" if TEST_SIZE > 0 else "  Test size:          FULL dataset")
print(f"  Loss factor (L1):   p{LOSS_FACTOR} (per-signal)")
print(f"  Time factor (L2):   p{TIME_FACTOR} (temporal)")
print(f"  Signal factor (L3): p{SIGNAL_FACTOR} (window)")
print(f"  Started at:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'

# Input paths
ATTACKS_DIR = DATA_DIR / 'attacks'
MODELS_DIR = RESULTS_DIR / 'training' / 'models'

# Output paths
DETECTION_DIR = RESULTS_DIR / 'detection'
DETECTION_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
WINDOW_LENGTH = 50
CHUNK_SIZE = 50000  # Process 50K timesteps at a time (memory-efficient!)
BATCH_SIZE = 1000   # Batch size for model prediction

# Signal names
SIGNAL_NAMES = [
    'pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle',
    'heading', 'cog', 'variation', 'sog', 'longitude',
    'rudder_angle_order', 'latitude', 'roll', 'rate_of_turn', 'rudder_position'
]
NUM_SIGNALS = len(SIGNAL_NAMES)

# ============================================================================
# Step 1: Load Attack Dataset
# ============================================================================

print("[1/7] Loading attack dataset...")
attack_file = ATTACKS_DIR / 'attack_dataset.npz'
attack_data = np.load(attack_file, allow_pickle=True)

timesteps_full = attack_data['X_data']  # (4.9M, 15)
labels_full = attack_data['y_labels']   # (4.9M,)
label_map = attack_data['label_map'].item()

# Use subset for testing or full dataset
if TEST_SIZE > 0:
    timesteps = timesteps_full[:TEST_SIZE]
    labels = labels_full[:TEST_SIZE]
    print(f"✓ Using SUBSET: {len(timesteps):,} timesteps (test mode)")
else:
    timesteps = timesteps_full
    labels = labels_full
    print(f"✓ Using FULL dataset: {len(timesteps):,} timesteps")

# Show distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"\nData distribution:")
for label_id, count in zip(unique, counts):
    label_name = label_map[int(label_id)]  # Convert numpy int to Python int
    pct = (count / len(labels)) * 100
    print(f"  {label_name:15s}: {count:10,} ({pct:5.2f}%)")
print()

# ============================================================================
# Step 2: Load Autoencoder
# ============================================================================

print("[2/7] Loading autoencoder (T=1)...")
model_path = MODELS_DIR / 'autoencoder_T1_best.h5'
model = tf.keras.models.load_model(model_path, compile=False)
print(f"✓ Model loaded: {model_path}")
print()

# ============================================================================
# Step 3: TIER 1 - Calculate Per-Signal Thresholds
# ============================================================================

print("[3/7] TIER 1: Calculating per-signal thresholds...")
print(f"  Using NORMAL data only (label=0) for threshold calculation")
print(f"  Processing in chunks of {CHUNK_SIZE:,} timesteps")
print()

# Collect errors for each signal from NORMAL data only
signal_errors_lists = [[] for _ in range(NUM_SIGNALS)]

normal_mask = (labels == 0)
normal_timesteps = timesteps[normal_mask]
print(f"  Normal data: {len(normal_timesteps):,} timesteps")

# Process normal data in chunks
num_normal = len(normal_timesteps)
for chunk_start in range(0, num_normal - WINDOW_LENGTH, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE + WINDOW_LENGTH, num_normal)
    chunk = normal_timesteps[chunk_start:chunk_end]
    
    # Create sliding windows
    num_windows = len(chunk) - WINDOW_LENGTH + 1
    if num_windows <= 0:
        continue
    
    windows = np.lib.stride_tricks.sliding_window_view(
        chunk, window_shape=(WINDOW_LENGTH,), axis=0
    ).transpose(0, 2, 1).astype(np.float32)
    
    # Process in batches
    for i in range(0, len(windows), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(windows))
        batch_windows = windows[i:batch_end]
        batch_input = batch_windows[..., np.newaxis]
        
        # Reconstruct
        reconstructed = model.predict(batch_input, verbose=0)
        
        # Calculate ABSOLUTE ERROR (like CANShield!)
        abs_error = np.abs(batch_input - reconstructed)[:, :, :, 0]
        
        # Store per-signal
        for sig_idx in range(NUM_SIGNALS):
            signal_errors_lists[sig_idx].append(abs_error[:, :, sig_idx].flatten())
    
    print(f"  Processed {chunk_end:,}/{num_normal:,} normal timesteps...")

# Calculate thresholds
signal_thresholds = np.zeros(NUM_SIGNALS)
print(f"\nSignal-level thresholds (p{LOSS_FACTOR}):")
for sig_idx in range(NUM_SIGNALS):
    all_errors = np.concatenate(signal_errors_lists[sig_idx])
    threshold = np.percentile(all_errors, LOSS_FACTOR)
    signal_thresholds[sig_idx] = threshold
    print(f"  Signal {sig_idx:2d} ({SIGNAL_NAMES[sig_idx]:20s}): {threshold:.6f}")

print()

# ============================================================================
# Step 4: TIER 2 - Calculate Temporal Thresholds
# ============================================================================

print("[4/7] TIER 2: Calculating temporal thresholds...")
print(f"  Counting anomalous timesteps per signal per window")
print()

# Collect temporal anomaly fractions for each signal
temporal_fractions_lists = [[] for _ in range(NUM_SIGNALS)]

# Process normal data again in chunks
for chunk_start in range(0, num_normal - WINDOW_LENGTH, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE + WINDOW_LENGTH, num_normal)
    chunk = normal_timesteps[chunk_start:chunk_end]
    
    num_windows = len(chunk) - WINDOW_LENGTH + 1
    if num_windows <= 0:
        continue
    
    windows = np.lib.stride_tricks.sliding_window_view(
        chunk, window_shape=(WINDOW_LENGTH,), axis=0
    ).transpose(0, 2, 1).astype(np.float32)
    
    # Process in batches
    for i in range(0, len(windows), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(windows))
        batch_windows = windows[i:batch_end]
        batch_input = batch_windows[..., np.newaxis]
        
        reconstructed = model.predict(batch_input, verbose=0)
        abs_error = np.abs(batch_input - reconstructed)[:, :, :, 0]
        
        # Apply signal thresholds → binary
        signal_anomalies = (abs_error > signal_thresholds).astype(int)
        
        # Count fraction of timesteps anomalous per signal
        temporal_fraction = np.sum(signal_anomalies, axis=1) / WINDOW_LENGTH
        
        # Store per-signal
        for sig_idx in range(NUM_SIGNALS):
            temporal_fractions_lists[sig_idx].append(temporal_fraction[:, sig_idx])
    
    print(f"  Processed {chunk_end:,}/{num_normal:,} normal timesteps...")

# Calculate temporal thresholds
temporal_thresholds = np.zeros(NUM_SIGNALS)
print(f"\nTemporal thresholds (p{TIME_FACTOR}):")
for sig_idx in range(NUM_SIGNALS):
    all_fractions = np.concatenate(temporal_fractions_lists[sig_idx])
    threshold = np.percentile(all_fractions, TIME_FACTOR)
    temporal_thresholds[sig_idx] = threshold
    print(f"  Signal {sig_idx:2d} ({SIGNAL_NAMES[sig_idx]:20s}): {threshold:.4f} ({threshold*100:.1f}% of timesteps)")

print()

# ============================================================================
# Step 5: TIER 3 - Calculate Final Threshold
# ============================================================================

print("[5/7] TIER 3: Calculating final window threshold...")
print(f"  Counting fraction of signals flagged per window")
print()

# Collect signal fractions for all normal windows
signal_fractions_list = []

# Process normal data one more time
for chunk_start in range(0, num_normal - WINDOW_LENGTH, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE + WINDOW_LENGTH, num_normal)
    chunk = normal_timesteps[chunk_start:chunk_end]
    
    num_windows = len(chunk) - WINDOW_LENGTH + 1
    if num_windows <= 0:
        continue
    
    windows = np.lib.stride_tricks.sliding_window_view(
        chunk, window_shape=(WINDOW_LENGTH,), axis=0
    ).transpose(0, 2, 1).astype(np.float32)
    
    for i in range(0, len(windows), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(windows))
        batch_windows = windows[i:batch_end]
        batch_input = batch_windows[..., np.newaxis]
        
        reconstructed = model.predict(batch_input, verbose=0)
        abs_error = np.abs(batch_input - reconstructed)[:, :, :, 0]
        
        # Tier 1: Signal anomalies
        signal_anomalies = (abs_error > signal_thresholds).astype(int)
        
        # Tier 2: Temporal fraction
        temporal_fraction = np.sum(signal_anomalies, axis=1) / WINDOW_LENGTH
        
        # Tier 2: Signal-level flags
        signal_flags = (temporal_fraction > temporal_thresholds).astype(int)
        
        # Tier 3: Fraction of signals flagged
        signal_fraction = np.sum(signal_flags, axis=1) / NUM_SIGNALS
        
        signal_fractions_list.append(signal_fraction)
    
    print(f"  Processed {chunk_end:,}/{num_normal:,} normal timesteps...")

# Calculate final threshold
all_signal_fractions = np.concatenate(signal_fractions_list)
final_threshold = np.percentile(all_signal_fractions, SIGNAL_FACTOR)

print(f"\nFinal window threshold (p{SIGNAL_FACTOR}): {final_threshold:.4f} ({final_threshold*100:.1f}% of signals)")
print()

# ============================================================================
# Step 6: Apply Detection to ALL Data
# ============================================================================

print("[6/7] Applying 3-tier detection to ALL data...")
print(f"  Processing {len(timesteps):,} timesteps...")
print()

all_predictions = []
all_window_labels = []

for chunk_start in range(0, len(timesteps) - WINDOW_LENGTH, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE + WINDOW_LENGTH, len(timesteps))
    chunk = timesteps[chunk_start:chunk_end]
    chunk_labels = labels[chunk_start:chunk_end]
    
    num_windows = len(chunk) - WINDOW_LENGTH + 1
    if num_windows <= 0:
        continue
    
    windows = np.lib.stride_tricks.sliding_window_view(
        chunk, window_shape=(WINDOW_LENGTH,), axis=0
    ).transpose(0, 2, 1).astype(np.float32)
    
    window_labels = chunk_labels[WINDOW_LENGTH-1:]
    
    chunk_predictions = []
    
    for i in range(0, len(windows), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(windows))
        batch_windows = windows[i:batch_end]
        batch_input = batch_windows[..., np.newaxis]
        
        reconstructed = model.predict(batch_input, verbose=0)
        abs_error = np.abs(batch_input - reconstructed)[:, :, :, 0]
        
        # TIER 1
        signal_anomalies = (abs_error > signal_thresholds).astype(int)
        
        # TIER 2
        temporal_fraction = np.sum(signal_anomalies, axis=1) / WINDOW_LENGTH
        signal_flags = (temporal_fraction > temporal_thresholds).astype(int)
        
        # TIER 3
        signal_fraction = np.sum(signal_flags, axis=1) / NUM_SIGNALS
        predictions = (signal_fraction > final_threshold).astype(int)
        
        chunk_predictions.extend(predictions)
    
    all_predictions.extend(chunk_predictions)
    all_window_labels.extend(window_labels)
    
    print(f"  Processed {chunk_end:,}/{len(timesteps):,} timesteps...")

all_predictions = np.array(all_predictions)
all_window_labels = np.array(all_window_labels)

print(f"\n✓ Detection complete: {len(all_predictions):,} windows processed")
print()

# ============================================================================
# Step 7: Calculate Metrics
# ============================================================================

print("[7/7] Calculating performance metrics...")

true_labels = (all_window_labels > 0).astype(int)

tp = np.sum((true_labels == 1) & (all_predictions == 1))
tn = np.sum((true_labels == 0) & (all_predictions == 0))
fp = np.sum((true_labels == 0) & (all_predictions == 1))
fn = np.sum((true_labels == 1) & (all_predictions == 0))

accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*80}")
print(f"RESULTS: {OUTPUT_SUFFIX}")
print(f"{'='*80}")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%")
print(f"{'='*80}")

# Per-attack statistics
print(f"\nPer-attack detection rates:")
for label_id in sorted(label_map.keys()):
    label_name = label_map[label_id]
    mask = (all_window_labels == label_id)
    total = np.sum(mask)
    if total > 0:
        detected = np.sum(all_predictions[mask])
        rate = (detected / total) * 100
        print(f"  {label_name:15s}: {detected:6,}/{total:6,} = {rate:5.2f}%")

# Save results
results = {
    "experiment": OUTPUT_SUFFIX,
    "method": "canshield_chunked",
    "timestamp": datetime.now().isoformat(),
    "configuration": {
        "test_size": TEST_SIZE if TEST_SIZE > 0 else len(timesteps_full),
        "loss_factor": LOSS_FACTOR,
        "time_factor": TIME_FACTOR,
        "signal_factor": SIGNAL_FACTOR,
        "chunk_size": CHUNK_SIZE,
        "window_length": WINDOW_LENGTH
    },
    "thresholds": {
        "signal_level": {name: float(th) for name, th in zip(SIGNAL_NAMES, signal_thresholds)},
        "temporal_level": {name: float(th) for name, th in zip(SIGNAL_NAMES, temporal_thresholds)},
        "final_level": float(final_threshold)
    },
    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    "per_attack_stats": {}
}

for label_id in sorted(label_map.keys()):
    label_name = label_map[label_id]
    mask = (all_window_labels == label_id)
    total = int(np.sum(mask))
    detected = int(np.sum(all_predictions[mask]))
    rate = float(detected / total) if total > 0 else 0
    results["per_attack_stats"][label_name] = {
        "total": total,
        "detected": detected,
        "rate": rate
    }

output_file = DETECTION_DIR / f'summary_{OUTPUT_SUFFIX}.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved: {output_file}")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
