# 🎯 PHASE 2: CNN AUTOENCODER TRAINING - COMPLETE PLAN

**Project**: Lightweight AI for Maritime CAN Intrusion Detection  
**Date**: 10 November 2025  
**Status**: Ready to implement (Phase 0 ✅ | Phase 1 ✅ completed)

---

## 📊 WHAT WE HAVE (FROM PHASE 1)

```
results/preprocessing/
├── windows/
│   ├── windows_T1.npy    → (98,893, 50, 15)  120.3 MB
│   ├── windows_T5.npy    → (19,740, 50, 15)   24.1 MB
│   ├── windows_T10.npy   → (9,846, 50, 15)    12.0 MB
│   ├── windows_T20.npy   → (4,899, 50, 15)     6.0 MB
│   └── windows_T50.npy   → (1,930, 50, 15)     2.4 MB
│
└── parameters/
    ├── norm_params_T1.csv
    ├── norm_params_T5.csv
    ├── norm_params_T10.csv
    ├── norm_params_T20.csv
    └── norm_params_T50.csv
```

**Total**: 135,308 training windows, normalized [0,1], ready for CNN

---

## 🎓 HOW CANSHIELD (ORIGINAL PAPER) DID IT

### **Key Implementation Details from CANShield-main/**

#### **1. Architecture (from `get_autoencoder.py`):**

```python
# THEIR APPROACH: 2D-CNN (treats time×signals as 2D image)
input_shape = (time_step, num_signals, 1)  # (50, 20, 1)

Encoder:
  ZeroPadding2D((2,2))           # Add padding
  Conv2D(32, (5,5)) → LeakyReLU  # Extract features
  MaxPooling2D((2,2))            # Downsample
  Conv2D(16, (5,5)) → LeakyReLU  
  MaxPooling2D((2,2))
  Conv2D(16, (3,3)) → LeakyReLU
  MaxPooling2D((2,2))            # Bottleneck

Decoder:
  Conv2D(16, (3,3)) → LeakyReLU
  UpSampling2D((2,2))            # Upsample
  Conv2D(16, (5,5)) → LeakyReLU
  UpSampling2D((2,2))
  Conv2D(32, (5,5)) → LeakyReLU
  UpSampling2D((2,2))
  Conv2D(1, (3,3), activation='sigmoid')
  Cropping2D()                   # Remove padding
```

**Why 2D-CNN?**
- Captures both **temporal patterns** (along time axis)
- Captures **signal correlations** (along signal axis)
- Original paper used 20 signals × 50 timesteps = "image-like" representation

#### **2. Training Configuration (from `syncan.yaml`):**

```yaml
# Hyperparameters they used:
max_epoch: 500
batch_size: 128
validation_split: 0.1  # 10% for validation
optimizer: Adam(lr=0.0002, beta_1=0.5, beta_2=0.99)
loss: MeanSquaredError (MSE)

# Multi-scale configuration:
time_steps: [50]              # Window length
sampling_periods: [1, 5, 10]  # Stride values (they use 3, we have 5!)
window_step_train: 10         # Skip windows during training (speed up)

# Early stopping:
patience: 10  # Stop if val_loss doesn't improve for 10 epochs
```

#### **3. Training Process (from `run_development_canshield.py`):**

```python
# Pseudo-code of their workflow:
for time_step in [50]:
    for sampling_period in [1, 5, 10]:
        
        # Build or load model
        autoencoder = get_autoencoder(time_step, num_signals)
        
        # Load training data
        for file in training_files:
            x_train, y_train = load_data(file)
            
            # Train incrementally on each file
            autoencoder.fit(
                x_train, x_train,  # Autoencoder: input = output
                epochs=500,
                batch_size=128,
                validation_split=0.1,
                callbacks=[EarlyStopping, ModelCheckpoint]
            )
        
        # Save final model
        model.save(f"autoencoder_{time_step}_{sampling_period}.h5")
```

**Key insight**: They train **incrementally** on multiple files (because SynCAN dataset is huge). We have **one unified file**, so we'll train **once per time scale**.

#### **4. Threshold Calculation (from `load_thresholds.py`):**

```python
# THREE-TIER THRESHOLD SYSTEM:

# Tier 1: Loss Threshold (per signal, per time scale)
for signal in range(num_signals):
    reconstruction_errors = compute_errors(signal)
    threshold = np.percentile(reconstruction_errors, loss_factor)
    # loss_factor ∈ {90, 91, ..., 99, 99.5, 99.99}

# Tier 2: Time Threshold (temporal consistency)
for window in windows:
    anomalous_signals = count_signals_above_loss_threshold(window)
    ratio = anomalous_signals / num_signals
time_threshold = np.percentile(ratios, time_factor)
# time_factor ∈ {90, 91, ..., 99, 99.5, 99.99}

# Tier 3: Signal Threshold (signal consistency)
signal_threshold = np.percentile(signal_counts, signal_factor)
# signal_factor ∈ {90, 91, ..., 99, 99.5, 99.99}
```

**Simplified for our use**:
- We'll use **95th percentile** for loss threshold (2σ equivalent)
- We'll use **99th percentile** for stricter detection (3σ equivalent)
- We'll add **99.5th percentile** for very rare anomalies

---

## 🚀 OUR PLAN (ALIGNED WITH CANSHIELD)

### **Phase 2 Goals:**

1. ✅ Train **5 CNN autoencoders** (T=1, 5, 10, 20, 50)
2. ✅ Compute **reconstruction error thresholds** (95th, 99th, 99.5th percentiles)
3. ✅ Save **trained models** for Phase 3 deployment
4. ✅ Generate **training visualizations** (loss curves, reconstructions)
5. ✅ Validate **model performance** on held-out validation set

---

## 📐 ARCHITECTURE DESIGN

### **Option A: 2D-CNN (CANShield approach) ✅ RECOMMENDED**

**Input shape**: `(batch, 50, 15, 1)`
- 50 timesteps
- 15 signals
- 1 channel (grayscale "image")

**Architecture**:
```python
def build_2d_cnn_autoencoder(time_step=50, num_signals=15):
    """
    2D-CNN autoencoder (CANShield architecture adapted for 15 signals)
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Conv2D, LeakyReLU, MaxPooling2D, UpSampling2D,
        ZeroPadding2D, Cropping2D
    )
    
    in_shape = (time_step, num_signals, 1)  # (50, 15, 1)
    
    autoencoder = Sequential()
    
    # ================== ENCODER ==================
    autoencoder.add(ZeroPadding2D((2, 2), input_shape=in_shape))
    
    # Layer 1
    autoencoder.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
    # Layer 2
    autoencoder.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
    # Layer 3 (Bottleneck)
    autoencoder.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    
    # ================== DECODER ==================
    # Layer 4
    autoencoder.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))
    
    # Layer 5
    autoencoder.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))
    
    # Layer 6
    autoencoder.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    autoencoder.add(LeakyReLU(alpha=0.2))
    autoencoder.add(UpSampling2D((2, 2)))
    
    # Output Layer
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    
    # Crop to original shape
    temp_shape = autoencoder.output_shape
    diff_time = temp_shape[1] - in_shape[0]
    diff_signal = temp_shape[2] - in_shape[1]
    
    top = diff_time // 2
    bottom = diff_time - top
    left = diff_signal // 2
    right = diff_signal - left
    
    autoencoder.add(Cropping2D(cropping=((top, bottom), (left, right))))
    
    return autoencoder
```

**Why this architecture?**
- ✅ **Proven**: CANShield paper achieved 99%+ accuracy
- ✅ **Spatial reasoning**: Learns signal correlations (e.g., pitch ↔ roll)
- ✅ **Temporal patterns**: Captures time-series dynamics
- ✅ **Lightweight**: Only ~50K parameters (runs on embedded systems)

---

### **Option B: 1D-CNN (Alternative - simpler)**

```python
def build_1d_cnn_autoencoder(time_step=50, num_signals=15):
    """
    1D-CNN autoencoder (treats each signal independently)
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Conv1D, LeakyReLU, MaxPooling1D, UpSampling1D, Flatten, Reshape
    )
    
    in_shape = (time_step, num_signals)  # (50, 15)
    
    autoencoder = Sequential()
    
    # Encoder
    autoencoder.add(Conv1D(32, 5, activation='relu', padding='same', 
                           input_shape=in_shape))
    autoencoder.add(MaxPooling1D(2, padding='same'))
    autoencoder.add(Conv1D(16, 3, activation='relu', padding='same'))
    autoencoder.add(MaxPooling1D(2, padding='same'))
    
    # Decoder
    autoencoder.add(Conv1D(16, 3, activation='relu', padding='same'))
    autoencoder.add(UpSampling1D(2))
    autoencoder.add(Conv1D(32, 5, activation='relu', padding='same'))
    autoencoder.add(UpSampling1D(2))
    
    # Output
    autoencoder.add(Conv1D(num_signals, 3, activation='sigmoid', padding='same'))
    
    return autoencoder
```

**Recommendation**: Use **Option A (2D-CNN)** to align with CANShield paper.

---

## 🛠️ IMPLEMENTATION PLAN

### **Directory Structure (NEW files to create)**

```
src/training/
├── __init__.py
├── autoencoder_builder.py    ← Build 2D-CNN architecture
├── trainer.py                ← Training loop + callbacks
└── threshold_calculator.py   ← Compute percentile thresholds

scripts/
└── 03_train_autoencoders.py  ← Main training script

results/training/
├── models/
│   ├── autoencoder_T1.h5
│   ├── autoencoder_T5.h5
│   ├── autoencoder_T10.h5
│   ├── autoencoder_T20.h5
│   └── autoencoder_T50.h5
│
├── thresholds/
│   ├── thresholds_T1.json    ← {p95, p99, p99.5 per signal}
│   ├── thresholds_T5.json
│   ├── thresholds_T10.json
│   ├── thresholds_T20.json
│   └── thresholds_T50.json
│
├── histories/
│   ├── history_T1.json       ← Training/val loss per epoch
│   └── ...
│
└── visualizations/
    ├── training_loss_T1.png
    ├── reconstruction_examples_T1.png
    ├── error_distribution_T1.png
    └── ...
```

---

## 📝 STEP-BY-STEP EXECUTION PLAN

### **STEP 1: Create `src/training/autoencoder_builder.py`**

```python
"""
Build 2D-CNN autoencoder architecture (CANShield style)
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, LeakyReLU, MaxPooling2D, UpSampling2D,
    ZeroPadding2D, Cropping2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

def build_2d_cnn_autoencoder(time_step=50, num_signals=15):
    # ... (full implementation as above)
    pass

def compile_autoencoder(autoencoder):
    """Compile with Adam optimizer and MSE loss"""
    opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.99)
    autoencoder.compile(
        loss=MeanSquaredError(), 
        optimizer=opt, 
        metrics=['mae']  # Mean Absolute Error for monitoring
    )
    return autoencoder
```

---

### **STEP 2: Create `src/training/trainer.py`**

```python
"""
Train autoencoder with early stopping and checkpointing
"""
import json
import numpy as np
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_autoencoder(
    autoencoder, 
    X_train, 
    output_dir,
    time_scale,
    epochs=500,
    batch_size=128,
    validation_split=0.1
):
    """
    Train autoencoder on training data
    
    Args:
        autoencoder: Compiled Keras model
        X_train: Training windows (N, 50, 15, 1)
        output_dir: Path to save outputs
        time_scale: T=1,5,10,20,50
        epochs: Max training epochs
        batch_size: Training batch size
        validation_split: Fraction for validation
    
    Returns:
        Trained model, training history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    checkpoint_path = output_dir / f'autoencoder_T{time_scale}_best.h5'
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    # Train (autoencoder: input = output!)
    print(f"\n{'='*80}")
    print(f"Training autoencoder for T={time_scale}")
    print(f"{'='*80}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print(f"Validation split: {validation_split*100:.1f}%")
    print()
    
    history = autoencoder.fit(
        X_train, X_train,  # Input = target (autoencoder!)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = output_dir / f'autoencoder_T{time_scale}.h5'
    autoencoder.save(final_model_path)
    print(f"\n✓ Model saved: {final_model_path}")
    
    # Save training history
    history_path = output_dir / f'history_T{time_scale}.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"✓ History saved: {history_path}")
    
    return autoencoder, history
```

---

### **STEP 3: Create `src/training/threshold_calculator.py`**

```python
"""
Compute reconstruction error thresholds (CANShield style)
"""
import json
import numpy as np
from pathlib import Path

def compute_reconstruction_errors(autoencoder, X_data):
    """
    Compute per-sample, per-signal reconstruction errors
    
    Args:
        autoencoder: Trained model
        X_data: Windows (N, 50, 15, 1)
    
    Returns:
        errors: (N, 50, 15) - absolute errors per timestep per signal
    """
    X_recon = autoencoder.predict(X_data, verbose=0)
    errors = np.abs(X_data - X_recon).squeeze()  # Remove channel dim
    return errors

def calculate_thresholds(errors, percentiles=[95, 99, 99.5]):
    """
    Calculate percentile thresholds per signal
    
    Args:
        errors: (N, 50, 15) reconstruction errors
        percentiles: List of percentile values
    
    Returns:
        Dict: {signal_name: {p95: val, p99: val, p99.5: val}}
    """
    num_signals = errors.shape[2]
    
    # Flatten temporal dimension (only keep signal dimension)
    # errors_per_signal: (N*50, 15)
    errors_flat = errors.reshape(-1, num_signals)
    
    thresholds = {}
    for signal_idx in range(num_signals):
        signal_errors = errors_flat[:, signal_idx]
        
        thresholds[f'signal_{signal_idx}'] = {
            f'p{p}': float(np.percentile(signal_errors, p))
            for p in percentiles
        }
    
    # Global thresholds (across all signals)
    thresholds['global'] = {
        f'p{p}': float(np.percentile(errors_flat, p))
        for p in percentiles
    }
    
    return thresholds

def save_thresholds(thresholds, output_dir, time_scale):
    """Save thresholds to JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    threshold_path = output_dir / f'thresholds_T{time_scale}.json'
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"✓ Thresholds saved: {threshold_path}")
    return threshold_path
```

---

### **STEP 4: Create `scripts/03_train_autoencoders.py`** (MAIN SCRIPT)

```python
"""
PHASE 2: TRAIN CNN AUTOENCODERS
================================

Train 5 separate autoencoders (one per time scale T=1,5,10,20,50)
Compute reconstruction error thresholds for anomaly detection

Input:  results/preprocessing/windows/windows_T*.npy
Output: results/training/models/autoencoder_T*.h5
        results/training/thresholds/thresholds_T*.json
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.autoencoder_builder import build_2d_cnn_autoencoder, compile_autoencoder
from training.trainer import train_autoencoder
from training.threshold_calculator import (
    compute_reconstruction_errors,
    calculate_thresholds,
    save_thresholds
)

def main():
    print("="*80)
    print("PHASE 2: CNN AUTOENCODER TRAINING")
    print("="*80)
    print()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    BASE_DIR = Path(__file__).parent.parent
    
    # Input (from Phase 1)
    WINDOWS_DIR = BASE_DIR / 'results' / 'preprocessing' / 'windows'
    
    # Output (Phase 2)
    MODELS_DIR = BASE_DIR / 'results' / 'training' / 'models'
    THRESHOLDS_DIR = BASE_DIR / 'results' / 'training' / 'thresholds'
    HISTORIES_DIR = BASE_DIR / 'results' / 'training' / 'histories'
    VIZ_DIR = BASE_DIR / 'results' / 'training' / 'visualizations'
    
    # Create directories
    for d in [MODELS_DIR, THRESHOLDS_DIR, HISTORIES_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Training config
    TIME_SCALES = [1, 5, 10, 20, 50]
    WINDOW_LENGTH = 50
    NUM_SIGNALS = 15
    
    # Hyperparameters (aligned with CANShield)
    MAX_EPOCHS = 500
    BATCH_SIZE = 128
    VALIDATION_SPLIT = 0.1
    
    # Threshold percentiles
    PERCENTILES = [95, 99, 99.5]
    
    print(f"✓ Input directory:  {WINDOWS_DIR}")
    print(f"✓ Output directory: {MODELS_DIR.parent}")
    print(f"✓ Time scales: {TIME_SCALES}")
    print(f"✓ Max epochs: {MAX_EPOCHS}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print()
    
    # ========================================================================
    # TRAIN AUTOENCODERS (ONE PER TIME SCALE)
    # ========================================================================
    
    for T in TIME_SCALES:
        print("\n" + "="*80)
        print(f"TIME SCALE T={T}")
        print("="*80)
        
        # ====================================================================
        # STEP 1: Load preprocessed windows
        # ====================================================================
        windows_path = WINDOWS_DIR / f'windows_T{T}.npy'
        
        if not windows_path.exists():
            print(f"❌ ERROR: {windows_path} not found!")
            continue
        
        print(f"\n[1/5] Loading windows from {windows_path.name}...")
        X_train = np.load(windows_path)
        
        print(f"   Loaded shape: {X_train.shape}")
        print(f"   Data range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"   Data mean: {X_train.mean():.4f}")
        
        # Reshape for 2D-CNN: (N, 50, 15) → (N, 50, 15, 1)
        X_train = X_train.reshape(-1, WINDOW_LENGTH, NUM_SIGNALS, 1)
        print(f"   Reshaped to: {X_train.shape}")
        
        # ====================================================================
        # STEP 2: Build autoencoder architecture
        # ====================================================================
        print(f"\n[2/5] Building 2D-CNN autoencoder...")
        autoencoder = build_2d_cnn_autoencoder(
            time_step=WINDOW_LENGTH,
            num_signals=NUM_SIGNALS
        )
        autoencoder = compile_autoencoder(autoencoder)
        
        print(f"\n   Model Summary:")
        autoencoder.summary()
        
        # ====================================================================
        # STEP 3: Train autoencoder
        # ====================================================================
        print(f"\n[3/5] Training autoencoder...")
        autoencoder, history = train_autoencoder(
            autoencoder=autoencoder,
            X_train=X_train,
            output_dir=MODELS_DIR,
            time_scale=T,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT
        )
        
        # ====================================================================
        # STEP 4: Compute reconstruction errors
        # ====================================================================
        print(f"\n[4/5] Computing reconstruction errors...")
        errors = compute_reconstruction_errors(autoencoder, X_train)
        
        print(f"   Error shape: {errors.shape}")
        print(f"   Error range: [{errors.min():.6f}, {errors.max():.6f}]")
        print(f"   Error mean: {errors.mean():.6f}")
        print(f"   Error std: {errors.std():.6f}")
        
        # ====================================================================
        # STEP 5: Calculate and save thresholds
        # ====================================================================
        print(f"\n[5/5] Calculating thresholds...")
        thresholds = calculate_thresholds(errors, percentiles=PERCENTILES)
        
        print(f"\n   Global thresholds:")
        for key, val in thresholds['global'].items():
            print(f"      {key}: {val:.6f}")
        
        save_thresholds(thresholds, THRESHOLDS_DIR, T)
        
        print(f"\n✅ Completed T={T}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2 TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✓ Trained {len(TIME_SCALES)} autoencoders")
    print(f"✓ Models saved to: {MODELS_DIR}")
    print(f"✓ Thresholds saved to: {THRESHOLDS_DIR}")
    print(f"\nNext: Run Phase 3 (Detection) to test on real intrusions!")
    print()

if __name__ == '__main__':
    main()
```

---

## 🔍 VERIFICATION CHECKLIST

After running training, verify:

```bash
# 1. Check models exist
ls -lh results/training/models/
# Should see: autoencoder_T1.h5, autoencoder_T5.h5, etc.

# 2. Check model sizes
du -sh results/training/models/*.h5
# Should be ~500KB - 2MB per model

# 3. Check thresholds
cat results/training/thresholds/thresholds_T1.json
# Should see: {"signal_0": {"p95": 0.023, "p99": 0.045, ...}}

# 4. Check training histories
cat results/training/histories/history_T1.json
# Should see: {"loss": [...], "val_loss": [...]}
```

---

## 📊 EXPECTED RESULTS

### **Training metrics (based on CANShield paper):**

| Time Scale | Training Loss (MSE) | Validation Loss | Training Time |
|------------|---------------------|-----------------|---------------|
| T=1 | ~0.001 - 0.002 | ~0.001 - 0.002 | ~30-60 min |
| T=5 | ~0.001 - 0.002 | ~0.001 - 0.002 | ~10-20 min |
| T=10 | ~0.001 - 0.002 | ~0.001 - 0.002 | ~5-10 min |
| T=20 | ~0.001 - 0.002 | ~0.001 - 0.002 | ~3-5 min |
| T=50 | ~0.001 - 0.002 | ~0.001 - 0.002 | ~1-2 min |

**Notes:**
- Lower time scales (T=1) have MORE data → longer training
- Early stopping should kick in around epoch 50-150
- Validation loss should closely track training loss (no overfitting)

### **Threshold values (example for T=1):**

```json
{
  "signal_0": {
    "p95": 0.0234,
    "p99": 0.0456,
    "p99.5": 0.0678
  },
  "signal_1": {
    "p95": 0.0189,
    "p99": 0.0342,
    "p99.5": 0.0501
  },
  ...
  "global": {
    "p95": 0.0298,
    "p99": 0.0512,
    "p99.5": 0.0723
  }
}
```

---

## 🎨 VISUALIZATION PLAN (OPTIONAL)

Add to training script:

```python
import matplotlib.pyplot as plt

def plot_training_curves(history, output_path):
    """Plot loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_examples(autoencoder, X_sample, output_path):
    """Show input vs reconstruction"""
    X_recon = autoencoder.predict(X_sample[:5])
    
    fig, axes = plt.subplots(5, 3, figsize=(12, 15))
    for i in range(5):
        # Input
        axes[i, 0].imshow(X_sample[i, :, :, 0].T, aspect='auto', cmap='viridis')
        axes[i, 0].set_title('Input')
        
        # Reconstruction
        axes[i, 1].imshow(X_recon[i, :, :, 0].T, aspect='auto', cmap='viridis')
        axes[i, 1].set_title('Reconstruction')
        
        # Error
        error = np.abs(X_sample[i] - X_recon[i])[:, :, 0]
        axes[i, 2].imshow(error.T, aspect='auto', cmap='Reds')
        axes[i, 2].set_title('Error')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 🚀 EXECUTION TIMELINE

```
Day 1 (2-3 hours):
├─ Create src/training/autoencoder_builder.py
├─ Create src/training/trainer.py
├─ Create src/training/threshold_calculator.py
└─ Test architecture builds correctly

Day 2 (1 hour):
├─ Create scripts/03_train_autoencoders.py
└─ Run training for T=1 (test run)

Day 3 (3-4 hours):
├─ Run full training (all 5 time scales)
├─ Validate outputs
└─ Generate visualizations

Total: ~1-2 days of work
```

---

## ✅ SUCCESS CRITERIA

Phase 2 is complete when:

- [x] 5 trained models saved (autoencoder_T1.h5 ... autoencoder_T50.h5)
- [x] 5 threshold files saved (thresholds_T1.json ... thresholds_T50.json)
- [x] Training loss < 0.005 for all models
- [x] Validation loss ≈ Training loss (no overfitting)
- [x] Models can reconstruct normal data with low error
- [x] Thresholds are sensible (p95 < p99 < p99.5)

---

## 🎯 NEXT PHASE

After Phase 2 completion → **Phase 3: Real-Time Detection**

Will use trained models to:
1. Load new CAN data
2. Reconstruct with autoencoders
3. Compare errors to thresholds
4. Detect intrusions!

---

**Ready to start coding?** 🚀

Let me know when you want to begin implementation!
