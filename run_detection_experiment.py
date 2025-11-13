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
data = attack_data['data']  # Shape: (N, 15, T)
labels = attack_data['attack_labels']  # Shape: (N,)

if MAX_SAMPLES:
    data = data[:MAX_SAMPLES]
    labels = labels[:MAX_SAMPLES]

num_samples = data.shape[0]
num_signals = data.shape[1]

print(f"✓ Loaded {num_samples:,} samples × {num_signals} signals")
print(f"  Normal: {np.sum(labels == 0):,}")
print(f"  Attacks: {np.sum(labels > 0):,}")
print()

# ============================================================================
# Load Autoencoders and Thresholds
# ============================================================================

print("[2/6] Loading trained autoencoders and thresholds...")
autoencoders = {}
thresholds = {}

for T in TIME_SCALES:
    # Load model
    model_path = MODELS_DIR / f'autoencoder_T{T}_best.h5'
    autoencoders[T] = tf.keras.models.load_model(model_path, compile=False)
    
    # Load thresholds
    threshold_file = THRESHOLDS_DIR / f'thresholds_T{T}.json'
    with open(threshold_file) as f:
        threshold_data = json.load(f)
        thresholds[T] = threshold_data['global'][THRESHOLD_PERCENTILE]
    
    print(f"✓ T={T:2d}: {THRESHOLD_PERCENTILE} threshold = {thresholds[T]:.6f}")

print()

# ============================================================================
# Create Windows for Each Time Scale
# ============================================================================

print("[3/6] Creating sliding windows for each time scale...")
windows_dict = {}

for T in TIME_SCALES:
    num_windows = (data.shape[2] - WINDOW_LENGTH) // T + 1
    windows = np.zeros((num_samples, num_windows, WINDOW_LENGTH, num_signals, 1))
    
    for i in range(num_windows):
        start_idx = i * T
        end_idx = start_idx + WINDOW_LENGTH
        if end_idx <= data.shape[2]:
            window_data = data[:, :, start_idx:end_idx]  # (N, 15, 50)
            windows[:, i, :, :, 0] = window_data.transpose(0, 2, 1)  # (N, 50, 15)
    
    windows_dict[T] = windows
    print(f"✓ T={T:2d}: {windows.shape}")

print()

# ============================================================================
# Run Detection
# ============================================================================

print("[4/6] Running detection with voting...")
print(f"  Batch size: {BATCH_SIZE}")
print()

# Store MSE errors for each autoencoder
all_mse_errors = {T: [] for T in TIME_SCALES}

for T in TIME_SCALES:
    print(f"Computing MSE for T={T}...")
    model = autoencoders[T]
    windows = windows_dict[T]
    
    # Process in batches
    num_windows = windows.shape[1]
    mse_per_window = []
    
    for w_idx in range(num_windows):
        window_batch = windows[:, w_idx, :, :, :]  # (N, 50, 15, 1)
        
        # Process in batches to avoid memory issues
        batch_mse = []
        for i in range(0, num_samples, BATCH_SIZE):
            batch = window_batch[i:i+BATCH_SIZE]
            reconstructed = model.predict(batch, verbose=0)
            mse = np.mean((batch - reconstructed) ** 2, axis=(1, 2, 3))
            batch_mse.append(mse)
        
        mse_per_window.append(np.concatenate(batch_mse))
    
    # Average MSE across all windows for each sample
    all_mse_errors[T] = np.mean(mse_per_window, axis=0)
    print(f"  ✓ MSE range: [{all_mse_errors[T].min():.6f}, {all_mse_errors[T].max():.6f}]")

print()

# ============================================================================
# Voting-based Detection
# ============================================================================

print("[5/6] Applying voting-based detection...")
print(f"  Voting threshold: {VOTING_THRESHOLD}/{len(TIME_SCALES)} autoencoders must detect anomaly")
print()

# Count votes for each sample
votes = np.zeros(num_samples, dtype=int)

for T in TIME_SCALES:
    detected = all_mse_errors[T] > thresholds[T]
    votes += detected.astype(int)
    print(f"  T={T:2d}: {np.sum(detected):6,}/{num_samples:,} flagged ({np.sum(detected)/num_samples*100:5.2f}%)")

# Final prediction: attack if >= VOTING_THRESHOLD autoencoders detected
predictions = (votes >= VOTING_THRESHOLD).astype(int)
num_detected = np.sum(predictions)

print()
print(f"Final detections: {num_detected:,}/{num_samples:,} ({num_detected/num_samples*100:.2f}%)")
print()

# ============================================================================
# Calculate Metrics
# ============================================================================

print("[6/6] Calculating performance metrics...")

# Confusion matrix
true_labels = (labels > 0).astype(int)  # 0=normal, 1=attack
tn = np.sum((true_labels == 0) & (predictions == 0))
fp = np.sum((true_labels == 0) & (predictions == 1))
fn = np.sum((true_labels == 1) & (predictions == 0))
tp = np.sum((true_labels == 1) & (predictions == 1))

# Metrics
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
    mask = (labels == attack_id)
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
    labels=labels,
    votes=votes,
    mse_errors={f'T{T}': all_mse_errors[T] for T in TIME_SCALES}
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
