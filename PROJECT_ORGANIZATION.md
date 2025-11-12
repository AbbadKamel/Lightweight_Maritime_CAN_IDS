# 🗂️ PROJECT ORGANIZATION - COMPLETE PIPELINE STRUCTURE

**Project**: Lightweight AI for Maritime CAN Intrusion Detection  
**Date**: 10 November 2025  
**Status**: Phase 0 ✅ | Phase 1 ✅ | Phase 2 ⏳ | Phase 3 ⏳

---

## 📁 DIRECTORY STRUCTURE

```
Lightweight_IA_V_2/
│
├── 📂 scripts/                          ← Entry point scripts (run these!)
│   ├── 00_validate_setup.py             ← Check environment is ready
│   ├── 01_initialize.py                 ← PHASE 0: Signal selection (EMPTY - not needed)
│   ├── 02_preprocess_data.py            ← PHASE 1: Create windows (EMPTY - not needed)
│   ├── decode_brute_frames.py           ← PHASE -1: Decode raw CAN frames
│   ├── merge_n2k_files.py               ← Utility: Merge multiple CAN logs
│   └── 00_generate_dummy_data.py        ← Testing: Create synthetic data
│
├── 📂 src/                              ← Core Python modules (imported by scripts)
│   ├── 📂 initialization/               ← PHASE 0 modules
│   │   ├── signal_selector.py           ← Remove low-quality signals
│   │   ├── correlation_analyzer.py      ← Compute Pearson correlation
│   │   └── signal_reorderer.py          ← Preserve signal order for CNN
│   │
│   ├── 📂 preprocessing/                ← PHASE 1 modules
│   │   ├── n2k_decoder.py               ← NMEA 2000 protocol decoder (pure Python)
│   │   ├── fifo_queue.py                ← Forward-fill missing values
│   │   ├── view_builder.py              ← Create multi-scale sliding windows
│   │   ├── normalizer.py                ← Min-max scaling [0,1]
│   │   └── __init__.py
│   │
│   ├── 📂 training/                     ← PHASE 2 modules (EMPTY - to be created)
│   │   └── (empty)
│   │
│   ├── 📂 detection/                    ← PHASE 3 modules (EMPTY - to be created)
│   │   └── (empty)
│   │
│   ├── 📂 models/                       ← CNN architectures
│   │   └── (to be created)
│   │
│   ├── 📂 evaluation/                   ← Performance metrics
│   │   └── (empty)
│   │
│   └── 📂 utils/                        ← Shared utilities
│       └── (helpers, constants, etc.)
│
├── 📂 results/                          ← All outputs organized by phase
│   │
│   ├── 📂 raw_frames/                   ← PHASE -1 output
│   │   └── brute_frames.csv             ← Raw CAN frames (154,161 frames)
│   │
│   ├── 📂 fixed_decoder_data/           ← PHASE -1 output
│   │   └── decoded_brute_frames.csv     ← Decoded messages (98,942 × 23 columns)
│   │
│   ├── 📂 initialization/               ← PHASE 0 outputs
│   │   ├── signal_order.txt             ← 15 selected signals IN ORDER
│   │   ├── correlation_matrix.csv       ← 15×15 Pearson correlations
│   │   ├── correlation_heatmap.png      ← Visual correlation matrix
│   │   ├── dendrogram.png               ← Hierarchical clustering
│   │   ├── data_quality_report.txt      ← Per-signal statistics
│   │   └── initialization_summary.txt   ← Human-readable summary
│   │
│   ├── 📂 preprocessing/                ← PHASE 1 outputs
│   │   ├── 📂 windows/                  ← Training-ready data
│   │   │   ├── windows_T1.npy           ← 98,893 windows (stride=1)
│   │   │   ├── windows_T5.npy           ← 19,740 windows (stride=5)
│   │   │   ├── windows_T10.npy          ← 9,846 windows (stride=10)
│   │   │   ├── windows_T20.npy          ← 4,899 windows (stride=20)
│   │   │   └── windows_T50.npy          ← 1,930 windows (stride=50)
│   │   │
│   │   ├── 📂 parameters/               ← Normalization configs
│   │   │   ├── norm_params_T1.csv       ← Min/max for T=1
│   │   │   ├── norm_params_T5.csv       ← Min/max for T=5
│   │   │   ├── norm_params_T10.csv      ← Min/max for T=10
│   │   │   ├── norm_params_T20.csv      ← Min/max for T=20
│   │   │   └── norm_params_T50.csv      ← Min/max for T=50
│   │   │
│   │   ├── 📂 visualizations/           ← Quality plots (10 PNG files)
│   │   │   ├── 01_real_signals_timeseries.png
│   │   │   ├── 02_normalization_effect.png
│   │   │   ├── 03_multiscale_*.png
│   │   │   ├── 04_distribution_*.png
│   │   │   └── 05_sample_window_heatmap.png
│   │   │
│   │   └── preprocessing_summary.txt    ← Phase 1 summary
│   │
│   ├── 📂 training/                     ← PHASE 2 outputs (to be created)
│   │   ├── 📂 models/
│   │   │   ├── autoencoder_T1.h5        ← Trained model for T=1
│   │   │   ├── autoencoder_T5.h5
│   │   │   ├── autoencoder_T10.h5
│   │   │   ├── autoencoder_T20.h5
│   │   │   └── autoencoder_T50.h5
│   │   │
│   │   ├── 📂 thresholds/
│   │   │   ├── thresholds_T1.json       ← Anomaly thresholds
│   │   │   └── ...
│   │   │
│   │   ├── 📂 training_history/
│   │   │   ├── history_T1.csv           ← Training loss curves
│   │   │   └── ...
│   │   │
│   │   └── 📂 visualizations/
│   │       ├── training_loss_T1.png
│   │       ├── reconstruction_error_distribution_T1.png
│   │       └── ...
│   │
│   └── 📂 detection/                    ← PHASE 3 outputs (to be created)
│       ├── 📂 test_results/
│       ├── 📂 attack_detection/
│       └── 📂 confusion_matrices/
│
├── 📂 CANShield/                        ← Reference implementation (original paper)
│   └── src/                             ← Contains modules we can reference
│
├── 📂 docs/                             ← Documentation
│   ├── PHASE_0_INITIALIZATION_DETAILED_WALKTHROUGH.md  ← ✅ Just created
│   └── PHASE_1_PREPROCESSING_DETAILED_WALKTHROUGH.md   ← ✅ Just created
│
├── 📂 config/                           ← Configuration files
│   └── (YAML configs for training hyperparameters)
│
├── 📂 models/                           ← Saved trained models
│   └── (will contain .h5 or .keras files)
│
├── 📂 notebooks/                        ← Jupyter notebooks for exploration
│   └── (interactive analysis)
│
├── 📂 tests/                            ← Unit tests
│   └── (pytest files)
│
├── 📜 run_preprocessing_REAL_DATA.py    ← **MAIN SCRIPT for Phase 1** ✅
├── 📜 requirements.txt                  ← Python dependencies
├── 📜 README.md                         ← Project overview
└── 📜 .gitignore                        ← Git ignore rules
```

---

## 🔄 PHASE EXECUTION FLOW

### **The ACTUAL workflow you've been following:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            COMPLETE PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

PHASE -1: DECODING (Raw CAN → Decoded Messages)
═══════════════════════════════════════════════
Script:  scripts/decode_brute_frames.py
Input:   results/raw_frames/brute_frames.csv (154,161 raw CAN frames)
Process: Parse NMEA 2000 protocol (PGN extraction, byte parsing)
Output:  results/fixed_decoder_data/decoded_brute_frames.csv
         (98,942 messages × 23 columns)
Status:  ✅ DONE

         ↓

PHASE 0: INITIALIZATION (Signal Quality Analysis)
═════════════════════════════════════════════════
Script:  Custom Python script (ad-hoc, not in scripts/)
Modules: src/initialization/
         - signal_selector.py
         - correlation_analyzer.py
         - signal_reorderer.py
Input:   results/fixed_decoder_data/decoded_brute_frames.csv
Process: 1. Check coverage (>95% threshold)
         2. Remove constant signals
         3. Remove duplicates (GNSS lat/lon)
         4. Compute correlation matrix
         5. Hierarchical clustering
         6. Select 15/23 signals
Output:  results/initialization/
         - signal_order.txt (15 signals)
         - correlation_heatmap.png
         - dendrogram.png
         - data_quality_report.txt
Status:  ✅ DONE
Docs:    PHASE_0_INITIALIZATION_DETAILED_WALKTHROUGH.md

         ↓

PHASE 1: PREPROCESSING (Multi-Scale Window Creation)
═════════════════════════════════════════════════════
Script:  run_preprocessing_REAL_DATA.py  ← **YOU RAN THIS!**
Modules: src/preprocessing/
         - fifo_queue.py (forward-fill)
         - view_builder.py (sliding windows)
         - normalizer.py (min-max scaling)
Input:   - decoded_brute_frames.csv
         - signal_order.txt
Process: 1. Load 15 selected signals
         2. Forward-fill missing values
         3. Create sliding windows:
            • T=1:  stride=1  → 98,893 windows
            • T=5:  stride=5  → 19,740 windows
            • T=10: stride=10 → 9,846 windows
            • T=20: stride=20 → 4,899 windows
            • T=50: stride=50 → 1,930 windows
         4. Normalize to [0,1]
         5. Save .npy files
         6. Generate visualizations
Output:  results/preprocessing/
         - windows/*.npy (5 files, 165 MB)
         - parameters/norm_params_T*.csv (5 files)
         - visualizations/*.png (10 plots)
Status:  ✅ DONE
Docs:    PHASE_1_PREPROCESSING_DETAILED_WALKTHROUGH.md

         ↓

PHASE 2: CNN TRAINING (Autoencoder Learning)
═════════════════════════════════════════════
Script:  scripts/03_train_autoencoders.py  ← **TO BE CREATED**
Modules: src/training/
         - autoencoder_builder.py  ← Build CNN architecture
         - trainer.py              ← Training loop
         - threshold_calculator.py ← Compute anomaly thresholds
Input:   results/preprocessing/windows/*.npy (5 files)
Process: For each time scale (T=1,5,10,20,50):
         1. Load windows
         2. Split train/validation (80/20)
         3. Build CNN autoencoder:
            • Encoder: Conv1D layers → bottleneck
            • Decoder: Conv1DTranspose → reconstruction
         4. Train with MSE loss
         5. Compute thresholds:
            • μ + 2σ (95% confidence)
            • μ + 3σ (99.7% confidence)
            • 99.5 percentile
         6. Save model (.h5 file)
         7. Plot training curves
Output:  results/training/
         - models/autoencoder_T*.h5 (5 models)
         - thresholds/thresholds_T*.json (5 JSON files)
         - training_history/history_T*.csv
         - visualizations/*.png
Status:  ⏳ NEXT STEP
Docs:    PHASE_2_TRAINING_DETAILED_WALKTHROUGH.md (to be created)

         ↓

PHASE 3: DETECTION (Real-Time Intrusion Detection)
═══════════════════════════════════════════════════
Script:  scripts/04_detect_intrusions.py  ← **TO BE CREATED**
Modules: src/detection/
         - online_detector.py      ← Real-time inference
         - threshold_checker.py    ← Compare errors to thresholds
         - alert_generator.py      ← Generate intrusion alerts
Input:   - Trained models (results/training/models/*.h5)
         - Thresholds (results/training/thresholds/*.json)
         - New CAN data (test set or live stream)
Process: 1. Load 5 trained autoencoders
         2. Load thresholds
         3. For each incoming window:
            • Forward pass through autoencoder
            • Compute reconstruction error
            • Compare to threshold
            • If error > threshold → ALERT!
         4. Multi-scale voting:
            • If 3/5 models detect anomaly → Intrusion confirmed
         5. Log detections
Output:  results/detection/
         - test_results.csv (per-window predictions)
         - attack_detection_report.txt
         - confusion_matrix.png
         - ROC_curves.png
Status:  ⏳ PENDING (after Phase 2)
Docs:    PHASE_3_DETECTION_DETAILED_WALKTHROUGH.md (to be created)
```

---

## 🎯 HOW PHASES ARE "SETTLED" (CURRENT STATE)

### **What you've actually executed:**

| Phase | Status | Script Used | Location | Output |
|-------|--------|-------------|----------|--------|
| **Phase -1** | ✅ DONE | `scripts/decode_brute_frames.py` | Uses `src/preprocessing/n2k_decoder.py` | `results/fixed_decoder_data/` |
| **Phase 0** | ✅ DONE | **Ad-hoc Python** (terminal commands) | Used `src/initialization/*` modules | `results/initialization/` |
| **Phase 1** | ✅ DONE | **`run_preprocessing_REAL_DATA.py`** (root dir) | Uses `src/preprocessing/*` | `results/preprocessing/` |
| **Phase 2** | ⏳ TODO | Not created yet | Will use `src/training/*` (empty) | `results/training/` |
| **Phase 3** | ⏳ TODO | Not created yet | Will use `src/detection/*` (empty) | `results/detection/` |

### **Why some scripts/ are empty?**

You noticed `scripts/01_initialize.py` and `scripts/02_preprocess_data.py` are **empty**. That's because:

1. **Phase 0 (Initialization)**: You ran it **manually via terminal** using Python REPL commands, not a script
2. **Phase 1 (Preprocessing)**: You used **`run_preprocessing_REAL_DATA.py`** in the **root directory**, not `scripts/02_preprocess_data.py`

### **What SHOULD happen (clean organization):**

**Option A: Move main script to scripts/ (recommended)**
```bash
mv run_preprocessing_REAL_DATA.py scripts/02_preprocess_data.py
```

**Option B: Keep root scripts + create wrappers**
```python
# scripts/02_preprocess_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_preprocessing_REAL_DATA import main
main()
```

---

## 📦 MODULE ORGANIZATION

### **src/initialization/** (Phase 0)

```python
# signal_selector.py
class SignalSelector:
    def remove_low_coverage(data, threshold=0.95)
    def remove_constant_signals(data)
    def remove_duplicates(data)

# correlation_analyzer.py
class CorrelationAnalyzer:
    def compute_correlation_matrix(data)
    def plot_heatmap(matrix)
    def hierarchical_clustering(matrix)

# signal_reorderer.py
def save_signal_order(signals, filepath)
def load_signal_order(filepath)
```

### **src/preprocessing/** (Phase 1)

```python
# n2k_decoder.py
class N2KDecoder:
    def decode_heading(pgn, data)
    def decode_attitude(pgn, data)
    def decode_rudder(pgn, data)
    # ... 10+ PGN decoders

# fifo_queue.py
class FIFOQueue:
    def forward_fill(data)
    def validate_no_nan(data)

# view_builder.py
class ViewBuilder:
    def create_sliding_windows(data, length=50, stride=1)
    def create_multi_scale_views(data, scales=[1,5,10,20,50])

# normalizer.py
class Normalizer:
    def fit(data)  # Compute min/max
    def transform(data)  # Apply normalization
    def save_params(filepath)  # Save for inference
```

### **src/training/** (Phase 2 - TO CREATE)

```python
# autoencoder_builder.py
class AutoencoderBuilder:
    def build_1d_cnn_autoencoder(input_shape)
    def compile_model(model, optimizer, loss)

# trainer.py
class AutoencoderTrainer:
    def train(model, train_data, val_data, epochs)
    def save_model(model, filepath)
    def plot_training_history(history)

# threshold_calculator.py
class ThresholdCalculator:
    def compute_reconstruction_errors(model, data)
    def calculate_statistical_thresholds(errors)
    def save_thresholds(thresholds, filepath)
```

### **src/detection/** (Phase 3 - TO CREATE)

```python
# online_detector.py
class OnlineDetector:
    def load_models(model_paths)
    def load_thresholds(threshold_paths)
    def predict_window(window)
    def multi_scale_voting(predictions)

# threshold_checker.py
class ThresholdChecker:
    def check_threshold(error, threshold)
    def generate_alert(window_id, time, signal)

# alert_generator.py
class AlertGenerator:
    def log_intrusion(detection_info)
    def send_alert(message)
```

---

## 🚀 NEXT STEPS TO ORGANIZE

### **Immediate actions:**

1. **Clean up script organization:**
   ```bash
   # Move Phase 1 script to scripts/
   mv run_preprocessing_REAL_DATA.py scripts/02_preprocess_data.py
   
   # Create Phase 0 script (consolidate terminal commands)
   # scripts/01_initialize.py
   ```

2. **Create Phase 2 training script:**
   ```bash
   # scripts/03_train_autoencoders.py
   ```

3. **Create Phase 3 detection script:**
   ```bash
   # scripts/04_detect_intrusions.py
   ```

4. **Populate src/training/ with modules:**
   ```
   src/training/
   ├── __init__.py
   ├── autoencoder_builder.py
   ├── trainer.py
   └── threshold_calculator.py
   ```

5. **Populate src/detection/ with modules:**
   ```
   src/detection/
   ├── __init__.py
   ├── online_detector.py
   ├── threshold_checker.py
   └── alert_generator.py
   ```

---

## 📊 CURRENT STATE SUMMARY

### **What exists and works:**

✅ **Data decoding**: `results/fixed_decoder_data/decoded_brute_frames.csv` (98,942 messages)  
✅ **Signal selection**: `results/initialization/signal_order.txt` (15 signals)  
✅ **Preprocessed windows**: `results/preprocessing/windows/*.npy` (135,308 windows, 165 MB)  
✅ **Normalization params**: `results/preprocessing/parameters/norm_params_T*.csv`  
✅ **Documentation**: Phase 0 & Phase 1 detailed walkthroughs  

### **What needs to be created:**

⏳ **Training modules**: `src/training/*`  
⏳ **Detection modules**: `src/detection/*`  
⏳ **Training script**: `scripts/03_train_autoencoders.py`  
⏳ **Detection script**: `scripts/04_detect_intrusions.py`  
⏳ **Phase 2 documentation**: `PHASE_2_TRAINING_DETAILED_WALKTHROUGH.md`  
⏳ **Phase 3 documentation**: `PHASE_3_DETECTION_DETAILED_WALKTHROUGH.md`  

---

## 🎓 KEY INSIGHT

**Your pipeline is NOT broken!** It's just **organically evolved**:

- Phase -1: Used `scripts/decode_brute_frames.py` ✅
- Phase 0: Ran **manually** (terminal commands) ✅
- Phase 1: Used **`run_preprocessing_REAL_DATA.py`** (root dir) ✅
- Phase 2: **Next to create**
- Phase 3: **Future work**

**All outputs are in the right place:** `results/*/`

**All modules are organized:** `src/*/`

**Just need to:**
1. Create Phase 2 & 3 scripts
2. Optionally reorganize existing scripts for consistency
3. Complete the pipeline!

---

**Ready to proceed with Phase 2 training?** 🚀
