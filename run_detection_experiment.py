#!/usr/bin/env python3
"""
Detection Experiment Runner - Run detection with specific parameters
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Parse command line arguments
if len(sys.argv) != 4:
    print("Usage: python run_detection_experiment.py <voting_threshold> <percentile> <output_suffix>")
    print("Example: python run_detection_experiment.py 2 p95 vote2_p95")
    sys.exit(1)

VOTING_THRESHOLD = int(sys.argv[1])
THRESHOLD_PERCENTILE = sys.argv[2]
OUTPUT_SUFFIX = sys.argv[3]

print("="*80)
print(f"DETECTION EXPERIMENT: {OUTPUT_SUFFIX}")
print("="*80)
print(f"  Voting threshold: {VOTING_THRESHOLD}/5 autoencoders")
print(f"  Threshold percentile: {THRESHOLD_PERCENTILE}")
print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# Import TensorFlow (suppress warnings)
import os
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
THRESHOLDS_DIR = RESULTS_DIR / 'training' / 'thresholds'

# Output paths
DETECTION_DIR = RESULTS_DIR / 'detection'
DETECTION_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
TIME_SCALES = [1, 5, 10, 20, 50]
WINDOW_LENGTH = 50
BATCH_SIZE = 1000
MAX_SAMPLES = None  # Process ALL samples (4.9M)

# ============================================================================
# Load Attack Dataset
# ============================================================================

print("[1/6] Loading attack dataset...")
attack_file = ATTACKS_DIR / 'attack_dataset.npz'
attack_data = np.load(attack_file)

# Get data and labels
timesteps = attack_data['X_data']  # Shape: (N, 15) - timesteps × signals
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

print("[2/6] Creating sliding windows...")
num_windows = num_timesteps - WINDOW_LENGTH + 1

if num_windows <= 0:
    print(f"❌ ERROR: Not enough timesteps ({num_timesteps}) to create windows of length {WINDOW_LENGTH}")
    sys.exit(1)

# Create sliding windows using stride tricks
windows = np.lib.stride_tricks.sliding_window_view(
    timesteps,
    window_shape=(WINDOW_LENGTH,),
    axis=0
)

# Reshape to [num_windows, window_length, num_signals]
windows = windows.transpose(0, 2, 1).astype(np.float32)

# Labels: use label at the END of each window
window_labels = labels[WINDOW_LENGTH-1:]

print(f"✓ Created {num_windows:,} windows of shape ({WINDOW_LENGTH}, {num_signals})")
print(f"  Normal: {np.sum(window_labels == 0):,}")
print(f"  Attacks: {np.sum(window_labels > 0):,}")
print()

num_samples = num_windows

# ============================================================================
# Load Autoencoders and Thresholds
# ============================================================================

print("[3/6] Loading trained autoencoders and thresholds...")
autoencoders = {}
thresholds = {}

for T in TIME_SCALES:
    # Load model
    model_path = MODELS_DIR / f'autoencoder_T{T}_best.h5'
    autoencoders[f'T{T}'] = tf.keras.models.load_model(model_path, compile=False)
    
    # Load thresholds
    threshold_file = THRESHOLDS_DIR / f'thresholds_T{T}.json'
    with open(threshold_file) as f:
        threshold_data = json.load(f)
        thresholds[f'T{T}'] = threshold_data['global'][THRESHOLD_PERCENTILE]
    
    print(f"✓ T={T:2d}: {THRESHOLD_PERCENTILE} threshold = {thresholds[f'T{T}']:.6f}")

print()

# ============================================================================
# Run Detection with Batch Processing
# ============================================================================

print("[4/6] Running detection with voting...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Voting threshold: {VOTING_THRESHOLD}/{len(TIME_SCALES)} autoencoders")
print()

# Store votes for each window
all_votes = np.zeros(num_windows, dtype=np.int32)

# Process each autoencoder
for T in TIME_SCALES:
    print(f"Computing MSE for T={T}...")
    model = autoencoders[f'T{T}']
    threshold = float(thresholds[f'T{T}'])
    
    # Process windows in batches
    num_detected = 0
    for i in range(0, num_windows, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, num_windows)
        batch_windows = windows[i:batch_end]
        
        # Add channel dimension: (N, 50, 15) -> (N, 50, 15, 1)
        batch_input = batch_windows[..., np.newaxis]
        
        # Reconstruct
        reconstructed = model.predict(batch_input, verbose=0)
        
        # Calculate MSE per window
        mse = np.mean(np.square(batch_input - reconstructed), axis=(1, 2, 3))
        
        # Vote: anomaly if MSE > threshold
        is_anomaly = (mse > threshold).astype(np.int32)
        all_votes[i:batch_end] += is_anomaly
        num_detected += np.sum(is_anomaly)
    
    print(f"  ✓ T={T:2d}: {num_detected:6,}/{num_windows:,} flagged ({num_detected/num_windows*100:5.2f}%)")

# Final prediction: attack if >= VOTING_THRESHOLD autoencoders detected
predictions = (all_votes >= VOTING_THRESHOLD).astype(np.int32)
num_detected_final = np.sum(predictions)

print()
print(f"Final detections: {num_detected_final:,}/{num_windows:,} ({num_detected_final/num_windows*100:.2f}%)")
print()

# ============================================================================
# Calculate Metrics
# ============================================================================

print("[5/6] Calculating performance metrics...")

# Confusion matrix
true_labels = (window_labels > 0).astype(int)  # 0=normal, 1=attack
tn = np.sum((true_labels == 0) & (predictions == 0))
fp = np.sum((true_labels == 0) & (predictions == 1))
fn = np.sum((true_labels == 1) & (predictions == 0))
tp = np.sum((true_labels == 1) & (predictions == 1))

# Metrics
num_samples = num_windows  # Update for window-based detection
accuracy = (tp + tn) / num_samples if num_samples > 0 else 0
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

print("[6/6] Saving results...")

# Save predictions
output_file = DETECTION_DIR / f'predictions_{OUTPUT_SUFFIX}.npz'
np.savez_compressed(
    output_file,
    predictions=predictions,
    labels=window_labels,
    votes=all_votes
)
print(f"✓ Predictions saved: {output_file}")

# Save summary
summary = {
    'experiment': OUTPUT_SUFFIX,
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'voting_threshold': VOTING_THRESHOLD,
        'threshold_percentile': THRESHOLD_PERCENTILE,
        'time_scales': TIME_SCALES,
        'num_samples': int(num_samples)
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
print(f"EXPERIMENT {OUTPUT_SUFFIX} COMPLETE!")
print("="*80)
print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%")
print("="*80)
