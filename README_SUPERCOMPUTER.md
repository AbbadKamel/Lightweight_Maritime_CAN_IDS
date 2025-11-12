# Maritime CAN IDS - Supercomputer Setup Guide

## 🖥️ Target Server: N315L-G17G01.ressource.unicaen.fr

**Specifications:**
- 2x Intel Xeon E5-2640 v4 (40 cores total)
- 256 GB RAM
- 8x GeForce GTX 1080 Ti (11 GB each)
- ~3.5 TB storage

---

## 📋 Quick Start (10 minutes)

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/Lightweight_IA_V_2.git
cd Lightweight_IA_V_2
```

### **2. Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### **3. Verify GPU Access**
```bash
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
# Expected: 8 GPUs detected
```

### **4. Download Dataset**
```bash
# Copy your decoded_brute_frames.csv to data/raw/
# OR download from your source
scp your_local_machine:/path/to/decoded_brute_frames.csv data/raw/
```

---

## 🚀 Full Pipeline Execution

### **Phase 0: Initialization** (30 seconds)
```bash
python3 scripts/00_initialize_project.py
```

### **Phase 1: Preprocessing** (2-3 minutes)
```bash
python3 scripts/01_preprocess_data.py
# Output: 98,942 messages → 135,000 windows
```

### **Phase 2: Training** (15-20 minutes on GPU)
```bash
python3 scripts/03_train_autoencoders.py
# Output: 5 autoencoders (T1, T5, T10, T20, T50)
# Expected val_loss: <3%, MAE: 6-10%
```

### **Phase 3: Attack Simulation & Detection**

**Step 1: Generate Attacks** (1 minute)
```bash
python3 scripts/05_simulate_attacks.py
# Output: 4.9M timesteps (70% normal, 30% attacks)
```

**Step 2: Run Detection** (30-60 minutes for FULL dataset)
```bash
# Edit scripts/06_test_detection_efficient.py:
# Set MAX_SAMPLES = None  (process all 4.9M samples)

python3 scripts/06_test_detection_efficient.py
```

**Step 3: Visualize Results**
```bash
python3 scripts/07_visualize_results.py
```

---

## ⚙️ Configuration for Supercomputer

### **Enable GPU Acceleration**

Edit `scripts/03_train_autoencoders.py` to use multiple GPUs:

```python
# At the top of the file, add:
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ {len(gpus)} GPUs available")
```

### **Parallel Training (Optional)**

Train each autoencoder on a different GPU simultaneously:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python3 scripts/03_train_autoencoders.py --time-scale 1 &

# Terminal 2  
CUDA_VISIBLE_DEVICES=1 python3 scripts/03_train_autoencoders.py --time-scale 5 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python3 scripts/03_train_autoencoders.py --time-scale 10 &

# Terminal 4
CUDA_VISIBLE_DEVICES=3 python3 scripts/03_train_autoencoders.py --time-scale 20 &

# Terminal 5
CUDA_VISIBLE_DEVICES=4 python3 scripts/03_train_autoencoders.py --time-scale 50 &
```

---

## 🔧 Retrain with Regularization (Option C)

To improve detection and reduce false positives:

```bash
# Edit scripts/03_train_autoencoders.py
# Change the model architecture to:

model.add(Dense(64, activation='relu', 
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
model.add(Dropout(0.3))

# Then retrain
python3 scripts/03_train_autoencoders.py
```

**Expected improvement:**
- Current: 98% recall, 30% precision
- After regularization: 85-90% recall, 70-80% precision

---

## 📊 Expected Results

### **With Current Settings (p95 threshold, 3/5 voting):**
```
Recall:    97.96%  ← Catches 98% of attacks
Precision: 30.17%  ← 70% false alarm rate
Accuracy:  31.43%
F1-Score:  46.13%

Per-Attack Performance:
  Flooding:    96.8%
  Suppress:    98.5%
  Plateau:     98.7%
  Continuous:  98.7%
  Playback:    97.1%
```

### **Memory Requirements:**
- Phase 1 (Preprocessing): ~2 GB RAM
- Phase 2 (Training): ~4 GB GPU memory per autoencoder
- Phase 3 (Detection): ~8 GB RAM (full dataset)

---

## 🐛 Troubleshooting

### **Out of Memory Error**
```bash
# Reduce batch size in detection script
# Edit scripts/06_test_detection_efficient.py
BATCH_SIZE = 500  # Instead of 1000
```

### **GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Verify TensorFlow sees GPUs
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### **CUDA Out of Memory**
```bash
# Train one autoencoder at a time
# Or reduce model size in scripts/03_train_autoencoders.py
```

---

## 📂 Project Structure

```
Lightweight_IA_V_2/
├── data/
│   ├── raw/                    # Place decoded_brute_frames.csv here
│   ├── attacks/               # Generated attack dataset
│   └── processed/             # Preprocessed windows
├── results/
│   ├── preprocessing/         # Normalized windows
│   ├── training/
│   │   ├── models/           # Trained autoencoders (.h5 files)
│   │   ├── thresholds/       # Detection thresholds (JSON)
│   │   └── histories/        # Training history
│   └── detection/            # Detection results
├── scripts/
│   ├── 00_initialize_project.py
│   ├── 01_preprocess_data.py
│   ├── 03_train_autoencoders.py
│   ├── 05_simulate_attacks.py
│   ├── 06_test_detection_efficient.py
│   └── 07_visualize_results.py
└── requirements.txt
```

---

## 📦 Required Files to Upload

**Before pushing to GitHub, ensure you have:**
1. ✅ `data/raw/decoded_brute_frames.csv` (98,942 messages)
2. ✅ All Python scripts in `scripts/`
3. ✅ `requirements.txt`
4. ✅ `.gitignore` (to exclude large files)
5. ✅ This README

**Large files to EXCLUDE (will be regenerated):**
- ❌ `results/training/models/*.h5` (500+ MB)
- ❌ `data/attacks/*.npz` (800+ MB)
- ❌ `results/preprocessing/windows/*.npy` (200+ MB)

---

## ⏱️ Estimated Time on Supercomputer

| Phase | Current PC | Supercomputer (GPU) | Speedup |
|-------|-----------|---------------------|---------|
| Preprocessing | 3 min | 1 min | 3x |
| Training (5 AE) | 30+ min | 5-8 min | 4-6x |
| Attack Simulation | 2 min | 30 sec | 4x |
| Detection (Full) | 3+ hours | 20-30 min | 6-9x |
| **TOTAL** | **~4 hours** | **~30-40 min** | **~6x faster** |

---

## 🎯 Next Steps

1. **Push to GitHub** (see commands below)
2. **SSH to supercomputer**
3. **Clone repository**
4. **Run full pipeline** (30-40 minutes)
5. **Download results** (predictions.npz, visualizations)

---

## 📝 Git Commands

```bash
# Initialize repository
cd "/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_2"
git init
git add .
git commit -m "Initial commit: Maritime CAN IDS with Multi-Scale Autoencoders"

# Create GitHub repository (on github.com)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/Lightweight_IA_V_2.git
git branch -M main
git push -u origin main
```

---

## ✅ Success Checklist

- [ ] Code pushed to GitHub
- [ ] SSH access to N315L-G17G01 confirmed
- [ ] Repository cloned on supercomputer
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] GPU access verified (8 GPUs)
- [ ] Dataset uploaded to `data/raw/`
- [ ] Phase 1 completed (preprocessing)
- [ ] Phase 2 completed (training with GPU)
- [ ] Phase 3 completed (detection on full dataset)
- [ ] Results downloaded locally

---

**Good luck! 🚀**
