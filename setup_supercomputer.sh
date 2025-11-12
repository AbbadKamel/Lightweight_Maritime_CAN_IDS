#!/bin/bash
# Supercomputer Setup Script for Maritime CAN IDS
# Run this script on N315L-G17G01.ressource.unicaen.fr after cloning

set -e  # Exit on error

echo "=================================================="
echo "Maritime CAN IDS - Supercomputer Setup"
echo "=================================================="
echo ""

# 1. Check Python version
echo "[1/6] Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Python 3 not found!"
    exit 1
fi
echo "✓ Python detected"
echo ""

# 2. Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# 3. Activate and install dependencies
echo "[3/6] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# 4. Check GPU access
echo "[4/6] Checking GPU access..."
python3 << 'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"✓ {len(gpus)} GPU(s) detected:")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu.name}")
if len(gpus) == 0:
    print("⚠️  WARNING: No GPUs detected! Training will be SLOW.")
EOF
echo ""

# 5. Create necessary directories
echo "[5/6] Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/attacks
mkdir -p results/preprocessing/windows
mkdir -p results/training/models
mkdir -p results/training/thresholds
mkdir -p results/training/histories
mkdir -p results/training/visualizations
mkdir -p results/detection
echo "✓ Directories created"
echo ""

# 6. Check for dataset
echo "[6/6] Checking for dataset..."
if [ -f "data/raw/decoded_brute_frames.csv" ]; then
    lines=$(wc -l < data/raw/decoded_brute_frames.csv)
    echo "✓ Dataset found: $lines lines"
else
    echo "⚠️  WARNING: Dataset not found at data/raw/decoded_brute_frames.csv"
    echo "   Please upload your dataset:"
    echo "   scp decoded_brute_frames.csv username@N315L-G17G01:/path/to/Lightweight_IA_V_2/data/raw/"
fi
echo ""

echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Upload dataset (if not done): scp decoded_brute_frames.csv ..."
echo "  2. Activate environment: source venv/bin/activate"
echo "  3. Run Phase 0: python3 scripts/00_initialize_project.py"
echo "  4. Run Phase 1: python3 scripts/01_preprocess_data.py"
echo "  5. Run Phase 2: python3 scripts/03_train_autoencoders.py"
echo "  6. Run Phase 3:"
echo "     - python3 scripts/05_simulate_attacks.py"
echo "     - python3 scripts/06_test_detection_efficient.py"
echo ""
echo "Estimated total time: 30-40 minutes"
echo ""
