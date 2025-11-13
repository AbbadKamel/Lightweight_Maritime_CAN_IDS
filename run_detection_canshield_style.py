#!/usr/bin/env python3
"""
CANShield-Style Hierarchical Detection
========================================

Implements 3-tier detection like CANShield paper:
1. Signal-level thresholds (per signal)
2. Time-level thresholds (% of timesteps anomalous)
3. Window-level thresholds (% of signals anomalous)

This should give MUCH better results than simple MSE averaging!
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import os

# Parse command line arguments
if len(sys.argv) != 5:
    print("Usage: python run_detection_canshield_style.py <loss_factor> <time_factor> <signal_factor> <output_suffix>")
    print("Example: python run_detection_canshield_style.py 95 95 95 canshield_95_95_95")
    print()
    print("Factors:")
    print("  loss_factor:   Percentile for per-signal reconstruction error (95, 99, etc.)")
    print("  time_factor:   Percentile for temporal anomaly count (95, 99, etc.)")
    print("  signal_factor: Percentile for multi-signal voting (95, 99, etc.)")
    sys.exit(1)

LOSS_FACTOR = float(sys.argv[1])
TIME_FACTOR = float(sys.argv[2])
SIGNAL_FACTOR = float(sys.argv[3])
OUTPUT_SUFFIX = sys.argv[4]

print("="*80)
print(f"CANSHIELD-STYLE HIERARCHICAL DETECTION: {OUTPUT_SUFFIX}")
print("="*80)
print(f"  Loss factor (signal-level):   p{LOSS_FACTOR}")
print(f"  Time factor (temporal-level): p{TIME_FACTOR}")
print(f"  Signal factor (window-level): p{SIGNAL_FACTOR}")
print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# Import TensorFlow (suppress warnings)
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
PREPROCESSING_DIR = RESULTS_DIR / 'preprocessing' / 'parameters'

# Output paths
DETECTION_DIR = RESULTS_DIR / 'detection'
DETECTION_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
TIME_SCALES = [1, 5, 10, 20, 50]
WINDOW_LENGTH = 50
BATCH_SIZE = 1000
MAX_SAMPLES = None  # Process ALL samples (4.9M)

# Signal names (from preprocessing)
SIGNAL_NAMES = [
    'pitch', 'depth', 'wind_speed', 'yaw', 'wind_angle',
    'heading', 'cog', 'variation', 'sog', 'longitude',
    'rudder_angle_order', 'latitude', 'roll', 'rate_of_turn', 'rudder_position'
]
NUM_SIGNALS = len(SIGNAL_NAMES)

# ============================================================================
# Load Attack Dataset
# ============================================================================

print("[1/7] Loading attack dataset...")
attack_file = ATTACKS_DIR / 'attack_dataset.npz'
attack_data = np.load(attack_file)

timesteps = attack_data['X_data']  # Shape: (N, 15)
labels = attack_data['y_labels']  # Shape: (N,)

if MAX_SAMPLES:
    timesteps = timesteps[:MAX_SAMPLES]
    labels = labels[:MAX_SAMPLES]

num_timesteps, num_signals = timesteps.shape

print(f"✓ Loaded {num_timesteps:,} timesteps × {num_signals} signals")
print()

# ============================================================================
# Create Sliding Windows
# ============================================================================

print("[2/7] Creating sliding windows...")
num_windows = num_timesteps - WINDOW_LENGTH + 1

if num_windows <= 0:
    print(f"❌ ERROR: Not enough timesteps")
    sys.exit(1)

# Create sliding windows
windows = np.lib.stride_tricks.sliding_window_view(
    timesteps,
    window_shape=(WINDOW_LENGTH,),
    axis=0
)
windows = windows.transpose(0, 2, 1).astype(np.float32)  # (num_windows, 50, 15)
window_labels = labels[WINDOW_LENGTH-1:]

print(f"✓ Created {num_windows:,} windows")
print()

# ============================================================================
# TIER 1: Calculate Per-Signal Reconstruction Errors
# ============================================================================

print("[3/7] TIER 1: Calculating per-signal reconstruction errors...")

# We'll use the best autoencoder (T=1) for now
# CANShield uses one model per time scale, we'll keep it simple
print("Loading autoencoder T=1...")
model = tf.keras.models.load_model(MODELS_DIR / 'autoencoder_T1_best.h5', compile=False)

# Process in batches and calculate ABSOLUTE ERROR per signal
all_signal_errors = np.zeros((num_windows, WINDOW_LENGTH, NUM_SIGNALS))

print("Processing windows in batches...")
for i in range(0, num_windows, BATCH_SIZE):
    batch_end = min(i + BATCH_SIZE, num_windows)
    batch_windows = windows[i:batch_end]
    
    # Add channel dimension
    batch_input = batch_windows[..., np.newaxis]  # (N, 50, 15, 1)
    
    # Reconstruct
    reconstructed = model.predict(batch_input, verbose=0)
    
    # Calculate ABSOLUTE ERROR (like CANShield), not MSE!
    abs_error = np.abs(batch_input - reconstructed)[:, :, :, 0]  # (N, 50, 15)
    
    all_signal_errors[i:batch_end] = abs_error
    
    if (i // BATCH_SIZE) % 100 == 0:
        print(f"  Processed {i:,}/{num_windows:,} windows...")

print(f"✓ Calculated errors for all windows")
print(f"  Shape: {all_signal_errors.shape} (windows × timesteps × signals)")
print()

# ============================================================================
# TIER 1: Compute Per-Signal Thresholds (Training Data)
# ============================================================================

print("[4/7] TIER 1: Computing per-signal thresholds...")

# Use only NORMAL data to set thresholds (like CANShield training phase)
normal_mask = (window_labels == 0)
normal_errors = all_signal_errors[normal_mask]

print(f"Using {np.sum(normal_mask):,} normal windows for threshold calculation")

# Compute threshold for each signal independently
signal_thresholds = np.zeros(NUM_SIGNALS)

for sig_idx in range(NUM_SIGNALS):
    # Flatten all normal errors for this signal across all windows/timesteps
    signal_errors_flat = normal_errors[:, :, sig_idx].flatten()
    
    # Compute percentile threshold
    threshold = np.percentile(signal_errors_flat, LOSS_FACTOR)
    signal_thresholds[sig_idx] = threshold
    
    print(f"  Signal {sig_idx:2d} ({SIGNAL_NAMES[sig_idx]:20s}): p{LOSS_FACTOR} = {threshold:.6f}")

print()

# ============================================================================
# TIER 2: Apply Signal Thresholds & Count Temporal Anomalies
# ============================================================================

print("[5/7] TIER 2: Applying signal thresholds and counting temporal anomalies...")

# For each window, for each signal, check if error > threshold
# Result: binary array (num_windows, 50, 15) - 1 if anomaly, 0 if normal
signal_anomalies = (all_signal_errors > signal_thresholds).astype(int)

# Count how many timesteps are anomalous per signal per window
# Result: (num_windows, 15) - fraction of timesteps that are anomalous
temporal_anomaly_fraction = np.sum(signal_anomalies, axis=1) / WINDOW_LENGTH

print(f"✓ Computed temporal anomaly fractions")
print(f"  Shape: {temporal_anomaly_fraction.shape} (windows × signals)")
print()

# ============================================================================
# TIER 2: Compute Temporal Thresholds
# ============================================================================

print("[6/7] TIER 2: Computing temporal thresholds...")

# Again, use only normal data
normal_temporal = temporal_anomaly_fraction[normal_mask]

# Compute threshold for each signal
temporal_thresholds = np.zeros(NUM_SIGNALS)

for sig_idx in range(NUM_SIGNALS):
    threshold = np.percentile(normal_temporal[:, sig_idx], TIME_FACTOR)
    temporal_thresholds[sig_idx] = threshold
    
    print(f"  Signal {sig_idx:2d} ({SIGNAL_NAMES[sig_idx]:20s}): p{TIME_FACTOR} = {threshold:.6f}")

print()

# ============================================================================
# TIER 3: Apply Temporal Thresholds & Multi-Signal Voting
# ============================================================================

print("[7/7] TIER 3: Applying temporal thresholds and multi-signal voting...")

# Binary: Is this signal anomalous in this window?
signal_level_anomalies = (temporal_anomaly_fraction > temporal_thresholds).astype(int)

# Count how many signals agree it's anomalous
# Result: (num_windows,) - fraction of signals that detected anomaly
signal_anomaly_fraction = np.sum(signal_level_anomalies, axis=1) / NUM_SIGNALS

# Compute final threshold using normal data
normal_signal_fraction = signal_anomaly_fraction[normal_mask]
final_threshold = np.percentile(normal_signal_fraction, SIGNAL_FACTOR)

print(f"Final threshold (p{SIGNAL_FACTOR}): {final_threshold:.6f}")
print()

# Final prediction: anomaly if signal_anomaly_fraction > threshold
predictions = (signal_anomaly_fraction > final_threshold).astype(int)

num_detected = np.sum(predictions)
print(f"Final detections: {num_detected:,}/{num_windows:,} ({num_detected/num_windows*100:.2f}%)")
print()

# ============================================================================
# Calculate Metrics
# ============================================================================

print("Calculating performance metrics...")

# Confusion matrix
true_labels = (window_labels > 0).astype(int)
tn = np.sum((true_labels == 0) & (predictions == 0))
fp = np.sum((true_labels == 0) & (predictions == 1))
fn = np.sum((true_labels == 1) & (predictions == 0))
tp = np.sum((true_labels == 1) & (predictions == 1))

# Metrics
accuracy = (tp + tn) / num_windows
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%")
print()

# Per-attack statistics
attack_names = ['Normal', 'Flooding', 'Suppress', 'Plateau', 'Continuous', 'Playback']
per_attack_stats = {}

for attack_id, attack_name in enumerate(attack_names):
    mask = (window_labels == attack_id)
    total = np.sum(mask)
    detected = np.sum(predictions[mask] == 1)
    rate = detected / total if total > 0 else 0
    
    per_attack_stats[attack_name] = {
        'total': int(total),
        'detected': int(detected),
        'rate': float(rate)
    }
    
    print(f"  {attack_name:15s}: {detected:7,}/{total:7,} = {rate*100:6.2f}%")

print()

# ============================================================================
# Save Results
# ============================================================================

print("Saving results...")

# Save predictions
output_file = DETECTION_DIR / f'predictions_{OUTPUT_SUFFIX}.npz'
np.savez_compressed(
    output_file,
    predictions=predictions,
    labels=window_labels,
    signal_anomaly_fraction=signal_anomaly_fraction,
    signal_thresholds=signal_thresholds,
    temporal_thresholds=temporal_thresholds,
    final_threshold=final_threshold
)
print(f"✓ Predictions saved: {output_file}")

# Save summary
summary = {
    'experiment': OUTPUT_SUFFIX,
    'method': 'canshield_hierarchical',
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'loss_factor': LOSS_FACTOR,
        'time_factor': TIME_FACTOR,
        'signal_factor': SIGNAL_FACTOR,
        'num_windows': int(num_windows),
        'num_signals': NUM_SIGNALS
    },
    'thresholds': {
        'signal_level': {SIGNAL_NAMES[i]: float(signal_thresholds[i]) for i in range(NUM_SIGNALS)},
        'temporal_level': {SIGNAL_NAMES[i]: float(temporal_thresholds[i]) for i in range(NUM_SIGNALS)},
        'final_level': float(final_threshold)
    },
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'confusion_matrix': [
        [int(tn), int(fp)],
        [int(fn), int(tp)]
    ],
    'per_attack_stats': per_attack_stats
}

summary_file = DETECTION_DIR / f'summary_{OUTPUT_SUFFIX}.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {summary_file}")

print()
print("="*80)
print(f"CANSHIELD-STYLE DETECTION COMPLETE!")
print("="*80)
print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%")
print("="*80)
