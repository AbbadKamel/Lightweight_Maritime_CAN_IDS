#!/usr/bin/env python3
"""
Phase 3 - Step 1: Attack Simulation Script
==========================================

This script simulates 5 types of CAN attacks on the normal dataset:
1. Flooding Attack - Repeat messages to saturate bus
2. Suppress Attack - Remove/block critical messages
3. Plateau Attack - Inject constant false values
4. Continuous Attack - Inject random changing values
5. Playback Attack - Replay old messages in wrong context

Input:  data/preprocessed/preprocessed_data.npz (100k normal samples)
Output: data/attacks/attack_dataset.npz (labeled with attack types)
"""

import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PREPROCESSED_DIR = DATA_DIR / 'processed'
ATTACKS_DIR = DATA_DIR / 'attacks'

# Attack distribution (percentages)
ATTACK_CONFIG = {
    'normal': 0.70,      # 70% normal data
    'flooding': 0.06,    # 6% flooding
    'suppress': 0.06,    # 6% suppress
    'plateau': 0.06,     # 6% plateau
    'continuous': 0.06,  # 6% continuous
    'playback': 0.06     # 6% playback
}

# Attack parameters (AGGRESSIVE but realistic - stay in [0,1] range!)
FLOODING_REPEAT = 100         # Repeat same message 100 times
SUPPRESS_RATIO = 0.9          # Remove 90% of messages (EXTREME!)
PLATEAU_VALUES = [0.0, 1.0]   # Alternate between 0 and 1 (square wave - unrealistic!)
PLATEAU_ATTACK_SIGNALS = 1.0  # Attack 100% of ALL signals
CONTINUOUS_ATTACK_SIGNALS = 1.0  # Attack 100% of ALL signals  
CONTINUOUS_PATTERN = 'square_wave'  # Generate square wave pattern (0→1→0→1...)
PLAYBACK_WINDOW = 200         # Larger replay segments

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Attack Functions
# ============================================================================

def flooding_attack(data, num_samples):
    """
    Flooding Attack: Repeat messages to saturate the bus
    
    Args:
        data: Normal data array
        num_samples: Number of attack samples to generate
    
    Returns:
        Attack data with flooding pattern
    """
    print(f"  Generating {num_samples} flooding attack samples...")
    
    attack_data = []
    samples_per_flood = FLOODING_REPEAT
    num_floods = num_samples // samples_per_flood
    
    # Select random messages to flood
    flood_indices = np.random.choice(len(data), size=num_floods, replace=False)
    
    for idx in flood_indices:
        # Repeat the same message multiple times
        repeated = np.tile(data[idx], (samples_per_flood, 1))
        attack_data.append(repeated)
    
    attack_data = np.vstack(attack_data)[:num_samples]  # Trim to exact size
    
    return attack_data


def suppress_attack(data, num_samples):
    """
    Suppress Attack: Remove/block critical messages (EXTREME - 80% suppression!)
    
    Args:
        data: Normal data array
        num_samples: Number of attack samples to generate
    
    Returns:
        Attack data with suppressed messages (filled with zeros)
    """
    print(f"  Generating {num_samples} suppress attack samples...")
    
    # Select segment from data
    start_idx = np.random.randint(0, len(data) - num_samples)
    attack_data = data[start_idx:start_idx + num_samples].copy()
    
    # AGGRESSIVELY suppress messages (80% set to zero!)
    num_suppress = int(num_samples * SUPPRESS_RATIO)
    suppress_indices = np.random.choice(num_samples, size=num_suppress, replace=False)
    attack_data[suppress_indices] = 0.0  # Suppressed messages = all zeros
    
    return attack_data


def plateau_attack(data, num_samples):
    """
    Plateau Attack: FLATLINE ATTACK - ALL signals set to ZERO! (EXTREME!)
    
    Args:
        data: Normal data array
        num_samples: Number of attack samples to generate
    
    Returns:
        Attack data with constant values (complete flatline)
    """
    print(f"  Generating {num_samples} plateau attack samples (FLATLINE - all zeros)...")
    
    # Select segment from data
    start_idx = np.random.randint(0, len(data) - num_samples)
    attack_data = data[start_idx:start_idx + num_samples].copy()
    
    # EXTREME: Replace ALL signals with ZERO (100% flatline!)
    num_signals = attack_data.shape[1]
    num_signals_to_attack = int(num_signals * PLATEAU_ATTACK_SIGNALS)
    
    signals_to_attack = np.random.choice(num_signals, size=num_signals_to_attack, replace=False)
    
    for signal_idx in signals_to_attack:
        plateau_value = np.random.choice(PLATEAU_VALUES)  # Will be 0.0
        attack_data[:, signal_idx] = plateau_value  # Complete flatline!
    
    return attack_data


def continuous_attack(data, num_samples):
    """
    Continuous Attack: Square wave pattern (0→1→0→1...) - VERY unrealistic!
    
    Args:
        data: Normal data array
        num_samples: Number of attack samples to generate
    
    Returns:
        Attack data with square wave pattern
    """
    print(f"  Generating {num_samples} continuous attack samples (SQUARE WAVE 0→1→0→1)...")
    
    # Select segment from data
    start_idx = np.random.randint(0, len(data) - num_samples)
    attack_data = data[start_idx:start_idx + num_samples].copy()
    
    # EXTREME: Replace ALL signals with square wave pattern (alternating 0 and 1)
    num_signals = attack_data.shape[1]
    num_signals_to_attack = int(num_signals * CONTINUOUS_ATTACK_SIGNALS)
    
    signals_to_attack = np.random.choice(num_signals, size=num_signals_to_attack, replace=False)
    
    for signal_idx in signals_to_attack:
        # Create square wave: [0, 1, 0, 1, 0, 1, ...]
        square_wave = np.tile([0.0, 1.0], num_samples // 2 + 1)[:num_samples]
        attack_data[:, signal_idx] = square_wave
    
    return attack_data


def playback_attack(data, num_samples):
    """
    Playback Attack: Record and replay old messages in wrong context
    
    Args:
        data: Normal data array
        num_samples: Number of attack samples to generate
    
    Returns:
        Attack data with replayed segments
    """
    print(f"  Generating {num_samples} playback attack samples...")
    
    attack_data = []
    num_replays = num_samples // PLAYBACK_WINDOW
    
    # Record segments from early in the dataset
    record_end = len(data) // 2  # Record from first half
    
    for _ in range(num_replays):
        # Record a segment
        record_start = np.random.randint(0, record_end - PLAYBACK_WINDOW)
        recorded_segment = data[record_start:record_start + PLAYBACK_WINDOW].copy()
        
        # Replay it (just use the recorded segment)
        attack_data.append(recorded_segment)
    
    attack_data = np.vstack(attack_data)[:num_samples]  # Trim to exact size
    
    return attack_data


# ============================================================================
# Main Attack Simulation
# ============================================================================

def simulate_all_attacks():
    """
    Main function to simulate all attack types on the normal dataset
    """
    print("\n" + "="*70)
    print("PHASE 3 - STEP 1: ATTACK SIMULATION")
    print("="*70)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create output directory
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. Load Preprocessed Data
    # ========================================================================
    print("\n[1/4] Loading preprocessed data...")
    
    # Load from Phase 1 preprocessing windows (T=1 has all timesteps)
    # This is the REAL BOAT DATA (98,942 messages from decoded_brute_frames.csv)
    windows_file = BASE_DIR / 'results' / 'preprocessing' / 'windows' / 'windows_T1_normalized.npy'
    signal_order_file = BASE_DIR / 'results' / 'initialization' / 'signal_order.txt'
    
    if not windows_file.exists():
        raise FileNotFoundError(f"Windows data not found: {windows_file}")
    
    # Load windows data (shape: [num_windows, num_signals, window_length, 1])
    windows_data = np.load(windows_file)
    
    # Flatten windows to individual timesteps for attack simulation
    # Shape: [num_windows, 15, 50, 1] -> [num_windows * 50, 15]
    num_windows, num_signals, window_length, _ = windows_data.shape
    X_data = windows_data.reshape(num_windows, num_signals, window_length).transpose(0, 2, 1).reshape(-1, num_signals)
    
    # Load signal names
    signal_names = []
    if signal_order_file.exists():
        with open(signal_order_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    signal_names.append(line)
    else:
        signal_names = [f'signal_{i}' for i in range(num_signals)]
    
    total_samples = len(X_data)
    
    print(f"  Loaded file: {windows_file}")
    print(f"  Original shape: {windows_data.shape} (windows)")
    print(f"  Flattened shape: {X_data.shape} (timesteps)")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Number of signals: {num_signals}")
    print(f"  Signal names: {', '.join(signal_names[:5])}...")
    
    # ========================================================================
    # 2. Calculate Attack Sample Counts
    # ========================================================================
    print("\n[2/4] Calculating attack distribution...")
    
    attack_counts = {}
    for attack_type, ratio in ATTACK_CONFIG.items():
        attack_counts[attack_type] = int(total_samples * ratio)
    
    # Adjust to ensure exact total
    total_generated = sum(attack_counts.values())
    if total_generated != total_samples:
        attack_counts['normal'] += (total_samples - total_generated)
    
    print("\n  Attack Distribution:")
    for attack_type, count in attack_counts.items():
        percentage = (count / total_samples) * 100
        print(f"    {attack_type.capitalize():12s}: {count:6,} samples ({percentage:5.2f}%)")
    print(f"    {'Total':12s}: {sum(attack_counts.values()):6,} samples")
    
    # ========================================================================
    # 3. Generate Attack Data
    # ========================================================================
    print("\n[3/4] Generating attack samples...")
    
    attack_datasets = {}
    attack_labels = {}
    
    # Label mapping: 0=Normal, 1=Flooding, 2=Suppress, 3=Plateau, 4=Continuous, 5=Playback
    label_map = {
        'normal': 0,
        'flooding': 1,
        'suppress': 2,
        'plateau': 3,
        'continuous': 4,
        'playback': 5
    }
    
    # Normal data
    print(f"\n  Preparing {attack_counts['normal']} normal samples...")
    normal_indices = np.random.choice(total_samples, size=attack_counts['normal'], replace=False)
    attack_datasets['normal'] = X_data[normal_indices]
    attack_labels['normal'] = np.zeros(attack_counts['normal'], dtype=np.int32)
    
    # Flooding attack
    if attack_counts['flooding'] > 0:
        print(f"\n  [Attack 1/5] Flooding Attack")
        attack_datasets['flooding'] = flooding_attack(X_data, attack_counts['flooding'])
        attack_labels['flooding'] = np.ones(attack_counts['flooding'], dtype=np.int32) * label_map['flooding']
    
    # Suppress attack
    if attack_counts['suppress'] > 0:
        print(f"\n  [Attack 2/5] Suppress Attack")
        attack_datasets['suppress'] = suppress_attack(X_data, attack_counts['suppress'])
        attack_labels['suppress'] = np.ones(attack_counts['suppress'], dtype=np.int32) * label_map['suppress']
    
    # Plateau attack
    if attack_counts['plateau'] > 0:
        print(f"\n  [Attack 3/5] Plateau Attack")
        attack_datasets['plateau'] = plateau_attack(X_data, attack_counts['plateau'])
        attack_labels['plateau'] = np.ones(attack_counts['plateau'], dtype=np.int32) * label_map['plateau']
    
    # Continuous attack
    if attack_counts['continuous'] > 0:
        print(f"\n  [Attack 4/5] Continuous Attack")
        attack_datasets['continuous'] = continuous_attack(X_data, attack_counts['continuous'])
        attack_labels['continuous'] = np.ones(attack_counts['continuous'], dtype=np.int32) * label_map['continuous']
    
    # Playback attack
    if attack_counts['playback'] > 0:
        print(f"\n  [Attack 5/5] Playback Attack")
        attack_datasets['playback'] = playback_attack(X_data, attack_counts['playback'])
        attack_labels['playback'] = np.ones(attack_counts['playback'], dtype=np.int32) * label_map['playback']
    
    # ========================================================================
    # 4. Combine and Shuffle All Data
    # ========================================================================
    print("\n[4/4] Combining and shuffling attack dataset...")
    
    # Combine all datasets
    all_data = np.vstack([attack_datasets[k] for k in attack_datasets.keys()])
    all_labels = np.hstack([attack_labels[k] for k in attack_labels.keys()])
    
    # Shuffle
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    
    print(f"  Combined dataset shape: {all_data.shape}")
    print(f"  Labels shape: {all_labels.shape}")
    
    # Verify label distribution
    print("\n  Label distribution after shuffling:")
    for attack_type, label in label_map.items():
        count = np.sum(all_labels == label)
        percentage = (count / len(all_labels)) * 100
        print(f"    {attack_type.capitalize():12s} (label={label}): {count:6,} samples ({percentage:5.2f}%)")
    
    # ========================================================================
    # 5. Save Attack Dataset
    # ========================================================================
    output_file = ATTACKS_DIR / 'attack_dataset.npz'
    
    print(f"\n[5/5] Saving attack dataset...")
    print(f"  Output file: {output_file}")
    
    np.savez_compressed(
        output_file,
        X_data=all_data,
        y_labels=all_labels,
        signal_names=signal_names,
        label_map=label_map,
        attack_counts=attack_counts
    )
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'total_samples': int(len(all_data)),
        'num_signals': int(num_signals),
        'label_map': {k: int(v) for k, v in label_map.items()},
        'attack_counts': {k: int(v) for k, v in attack_counts.items()},
        'attack_config': ATTACK_CONFIG,
        'attack_parameters': {
            'flooding_repeat': FLOODING_REPEAT,
            'suppress_ratio': SUPPRESS_RATIO,
            'plateau_values': PLATEAU_VALUES,
            'playback_window': PLAYBACK_WINDOW
        },
        'random_seed': RANDOM_SEED
    }
    
    metadata_file = ATTACKS_DIR / 'attack_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved: {metadata_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("ATTACK SIMULATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples: {len(all_data):,}")
    print(f"  Normal samples: {attack_counts['normal']:,} (70%)")
    print(f"  Attack samples: {len(all_data) - attack_counts['normal']:,} (30%)")
    print(f"\n🎯 Attack Types Generated:")
    print(f"  1. Flooding:   {attack_counts['flooding']:,} samples")
    print(f"  2. Suppress:   {attack_counts['suppress']:,} samples")
    print(f"  3. Plateau:    {attack_counts['plateau']:,} samples")
    print(f"  4. Continuous: {attack_counts['continuous']:,} samples")
    print(f"  5. Playback:   {attack_counts['playback']:,} samples")
    print(f"\n💾 Output Files:")
    print(f"  Attack dataset: {output_file}")
    print(f"  Metadata: {metadata_file}")
    print(f"\n✅ Ready for Phase 3 - Step 2: Detection System")
    print("="*70 + "\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    try:
        simulate_all_attacks()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
