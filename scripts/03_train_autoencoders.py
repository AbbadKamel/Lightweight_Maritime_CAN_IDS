"""
PHASE 2: CNN AUTOENCODER TRAINING
==================================

Train 5 separate CNN autoencoders (one per time scale T=1,5,10,20,50).
Compute reconstruction error thresholds for anomaly detection.

Input:
    - results/preprocessing/windows/windows_T*.npy (from Phase 1)
    - results/initialization/signal_order.txt (signal names)

Output:
    - results/training/models/autoencoder_T*.h5 (trained models)
    - results/training/thresholds/thresholds_T*.json (detection thresholds)
    - results/training/histories/history_T*.json (training curves)

Aligned with CANShield paper architecture and training strategy.

Author: Maritime CAN Intrusion Detection Pipeline
Date: November 2025
"""

import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.autoencoder_builder import build_2d_cnn_autoencoder, compile_autoencoder
from training.trainer import train_autoencoder
from training.threshold_calculator import (
    compute_reconstruction_errors,
    calculate_thresholds,
    save_thresholds
)


def load_signal_order(signal_file):
    """Load signal names from signal_order.txt"""
    signals = []
    with open(signal_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                signals.append(line)
    return signals


def main():
    """Main training pipeline"""
    
    print("="*80)
    print("PHASE 2: CNN AUTOENCODER TRAINING")
    print("="*80)
    print()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    BASE_DIR = Path(__file__).parent.parent
    
    # Input directories (from Phase 1)
    WINDOWS_DIR = BASE_DIR / 'results' / 'preprocessing' / 'windows'
    SIGNAL_ORDER_FILE = BASE_DIR / 'results' / 'initialization' / 'signal_order.txt'
    
    # Output directories (Phase 2)
    OUTPUT_DIR = BASE_DIR / 'results' / 'training'
    MODELS_DIR = OUTPUT_DIR / 'models'
    THRESHOLDS_DIR = OUTPUT_DIR / 'thresholds'
    HISTORIES_DIR = OUTPUT_DIR / 'histories'
    VIZ_DIR = OUTPUT_DIR / 'visualizations'
    
    # Create output directories
    for d in [MODELS_DIR, THRESHOLDS_DIR, HISTORIES_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    TIME_SCALES = [1, 5, 10, 20, 50]
    WINDOW_LENGTH = 50
    NUM_SIGNALS = 15
    
    # Hyperparameters (aligned with CANShield)
    MAX_EPOCHS = 500
    BATCH_SIZE = 128
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 0.0002
    
    # Threshold percentiles
    PERCENTILES = [95, 99, 99.5]
    
    # ========================================================================
    # VERIFY INPUTS
    # ========================================================================
    
    print("Configuration:")
    print("-" * 80)
    print(f"  Input directory:     {WINDOWS_DIR}")
    print(f"  Output directory:    {OUTPUT_DIR}")
    print(f"  Time scales:         {TIME_SCALES}")
    print(f"  Window length:       {WINDOW_LENGTH}")
    print(f"  Number of signals:   {NUM_SIGNALS}")
    print()
    
    print("Hyperparameters:")
    print("-" * 80)
    print(f"  Max epochs:          {MAX_EPOCHS}")
    print(f"  Batch size:          {BATCH_SIZE}")
    print(f"  Validation split:    {VALIDATION_SPLIT*100:.0f}%")
    print(f"  Learning rate:       {LEARNING_RATE}")
    print(f"  Early stop patience: 10 epochs")
    print()
    
    print("Threshold percentiles:")
    print("-" * 80)
    print(f"  {PERCENTILES}")
    print()
    
    # Verify signal order file exists
    if not SIGNAL_ORDER_FILE.exists():
        print(f"❌ ERROR: Signal order file not found: {SIGNAL_ORDER_FILE}")
        print("   Please run Phase 0 (Initialization) first!")
        return
    
    # Load signal names
    signal_names = load_signal_order(SIGNAL_ORDER_FILE)
    print(f"Signal names loaded ({len(signal_names)} signals):")
    print("-" * 80)
    for i, name in enumerate(signal_names, 1):
        print(f"  {i:2d}. {name}")
    print()
    
    if len(signal_names) != NUM_SIGNALS:
        print(f"⚠️  WARNING: Expected {NUM_SIGNALS} signals, got {len(signal_names)}")
        NUM_SIGNALS = len(signal_names)
        print(f"   Adjusting to {NUM_SIGNALS} signals")
        print()
    
    # ========================================================================
    # TRAINING LOOP (ONE AUTOENCODER PER TIME SCALE)
    # ========================================================================
    
    training_summary = []
    
    for T in TIME_SCALES:
        print("\n" + "="*80)
        print(f"TIME SCALE T={T}")
        print("="*80)
        
        # ====================================================================
        # STEP 1: Load preprocessed windows
        # ====================================================================
        
        windows_path = WINDOWS_DIR / f'windows_T{T}_normalized.npy'
        
        if not windows_path.exists():
            print(f"\n❌ ERROR: Windows file not found: {windows_path}")
            print(f"   Please run Phase 1 (Preprocessing) first!")
            continue
        
        print(f"\n[1/5] Loading preprocessed windows...")
        print(f"      File: {windows_path.name}")
        
        X_train = np.load(windows_path)
        
        print(f"      Loaded shape:  {X_train.shape}")
        print(f"      Data range:    [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"      Data mean:     {X_train.mean():.4f}")
        print(f"      Data std:      {X_train.std():.4f}")
        
        # Transpose if needed: (N, 15, 50, 1) → (N, 50, 15, 1)
        if X_train.shape[1] == NUM_SIGNALS and X_train.shape[2] == WINDOW_LENGTH:
            X_train = np.transpose(X_train, (0, 2, 1, 3))
            print(f"      Transposed to: {X_train.shape} (time×signals×channel)")
        # Reshape for 2D-CNN: (N, 50, 15) → (N, 50, 15, 1)
        elif X_train.ndim == 3:
            X_train = X_train.reshape(-1, WINDOW_LENGTH, NUM_SIGNALS, 1)
            print(f"      Reshaped to:   {X_train.shape} (added channel dimension)")
        
        # ====================================================================
        # STEP 2: Build autoencoder architecture
        # ====================================================================
        
        print(f"\n[2/5] Building 2D-CNN autoencoder...")
        
        autoencoder = build_2d_cnn_autoencoder(
            time_step=WINDOW_LENGTH,
            num_signals=NUM_SIGNALS
        )
        
        autoencoder = compile_autoencoder(
            autoencoder,
            learning_rate=LEARNING_RATE
        )
        
        # Print model summary (only for first time scale)
        if T == TIME_SCALES[0]:
            print()
            print("="*80)
            print("MODEL ARCHITECTURE (2D-CNN Autoencoder)")
            print("="*80)
            autoencoder.summary()
            print("="*80)
            
            total_params = autoencoder.count_params()
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Model size estimate: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        else:
            total_params = autoencoder.count_params()
            print(f"      Total params:  {total_params:,}")
        
        # ====================================================================
        # STEP 3: Train autoencoder
        # ====================================================================
        
        print(f"\n[3/5] Training autoencoder...")
        
        trained_model, history = train_autoencoder(
            autoencoder=autoencoder,
            X_train=X_train,
            output_dir=MODELS_DIR,
            time_scale=T,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=1
        )
        
        # ====================================================================
        # STEP 4: Compute reconstruction errors
        # ====================================================================
        
        print(f"\n[4/5] Computing reconstruction errors...")
        
        errors = compute_reconstruction_errors(
            autoencoder=trained_model,
            X_data=X_train,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # ====================================================================
        # STEP 5: Calculate and save thresholds
        # ====================================================================
        
        print(f"\n[5/5] Calculating thresholds...")
        
        thresholds = calculate_thresholds(
            errors=errors,
            signal_names=signal_names,
            percentiles=PERCENTILES
        )
        
        threshold_path = save_thresholds(
            thresholds=thresholds,
            output_dir=THRESHOLDS_DIR,
            time_scale=T,
            verbose=1
        )
        
        # ====================================================================
        # SUMMARY FOR THIS TIME SCALE
        # ====================================================================
        
        summary = {
            'time_scale': T,
            'windows': X_train.shape[0],
            'epochs_trained': history['metadata']['total_epochs'],
            'best_epoch': history['metadata']['best_epoch'],
            'final_loss': history['metadata']['final_loss'],
            'final_val_loss': history['metadata']['final_val_loss'],
            'best_val_loss': history['metadata']['best_val_loss'],
            'mean_error': thresholds['statistics']['mean_error'],
            'threshold_p95': thresholds['global']['p95'],
            'threshold_p99': thresholds['global']['p99'],
            'threshold_p99.5': thresholds['global']['p99.5']
        }
        
        training_summary.append(summary)
        
        print("\n" + "="*80)
        print(f"✅ COMPLETED T={T}")
        print("="*80)
        print(f"  Windows processed:   {summary['windows']:,}")
        print(f"  Epochs trained:      {summary['epochs_trained']}")
        print(f"  Best epoch:          {summary['best_epoch']}")
        print(f"  Final loss:          {summary['final_loss']:.6f}")
        print(f"  Final val_loss:      {summary['final_val_loss']:.6f}")
        print(f"  Best val_loss:       {summary['best_val_loss']:.6f}")
        print(f"  Mean error:          {summary['mean_error']:.6f}")
        print(f"  Threshold (p99):     {summary['threshold_p99']:.6f}")
        print("="*80)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("PHASE 2 TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\nTrained {len(training_summary)} autoencoders:")
    print("-" * 80)
    print(f"{'T':>3} {'Windows':>10} {'Epochs':>7} {'Best':>5} {'Loss':>10} {'Val Loss':>10} {'p99':>10}")
    print("-" * 80)
    
    for s in training_summary:
        print(
            f"{s['time_scale']:3d} "
            f"{s['windows']:10,} "
            f"{s['epochs_trained']:7d} "
            f"{s['best_epoch']:5d} "
            f"{s['final_loss']:10.6f} "
            f"{s['final_val_loss']:10.6f} "
            f"{s['threshold_p99']:10.6f}"
        )
    
    print("-" * 80)
    
    print(f"\nOutputs saved:")
    print("-" * 80)
    print(f"  Models:      {MODELS_DIR}")
    print(f"  Thresholds:  {THRESHOLDS_DIR}")
    print(f"  Histories:   {HISTORIES_DIR}")
    
    print(f"\nFiles created:")
    print("-" * 80)
    
    # List models
    models = sorted(MODELS_DIR.glob('autoencoder_T*.h5'))
    print(f"  Models ({len(models)}):")
    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"    - {model.name} ({size_mb:.1f} MB)")
    
    # List thresholds
    thresholds_files = sorted(THRESHOLDS_DIR.glob('thresholds_T*.json'))
    print(f"\n  Thresholds ({len(thresholds_files)}):")
    for thresh in thresholds_files:
        size_kb = thresh.stat().st_size / 1024
        print(f"    - {thresh.name} ({size_kb:.1f} KB)")
    
    # List histories
    histories = sorted(HISTORIES_DIR.glob('history_T*.json'))
    print(f"\n  Histories ({len(histories)}):")
    for hist in histories:
        size_kb = hist.stat().st_size / 1024
        print(f"    - {hist.name} ({size_kb:.1f} KB)")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("  ✓ Phase 0: Initialization    [DONE]")
    print("  ✓ Phase 1: Preprocessing     [DONE]")
    print("  ✓ Phase 2: Training          [DONE]")
    print("  → Phase 3: Detection         [NEXT]")
    print()
    print("Run Phase 3 to test intrusion detection on new CAN data!")
    print("="*80)
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Partial results may be saved in results/training/")
    except Exception as e:
        print(f"\n\n❌ ERROR during training:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
