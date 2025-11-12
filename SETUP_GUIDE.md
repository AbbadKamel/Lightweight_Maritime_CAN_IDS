# 🚀 SETUP GUIDE - Do This BEFORE Collecting Data

## ✅ Step-by-Step Setup (No Data Needed Yet!)

### **Step 1: Environment Setup** (5 minutes)

```bash
# Navigate to project
cd "/home/abbad241/Desktop/PhD/Journals_Articles_Papers/Next paper/Lightweight_IA_V_2"

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv)

# Install dependencies
pip install -r requirements.txt

# This will install:
# - TensorFlow, Keras (for deep learning)
# - Pandas, NumPy (for data processing)
# - Scikit-learn (for preprocessing)
# - Matplotlib, Seaborn (for visualization)
# - YAML (for configuration)
```

---

### **Step 2: Validate Installation** (2 minutes)

```bash
# Run validation script
python scripts/00_validate_setup.py
```

**Expected output:**
```
✅ All dependencies installed correctly!
✅ All project modules load successfully!
✅ All configuration files valid!
✅ All required directories exist!
✅ All modules function correctly!

🎉 ALL TESTS PASSED!
```

**If you see errors:**
- Missing packages? → `pip install -r requirements.txt`
- Import errors? → Check you're in project root
- TensorFlow issues? → `pip install tensorflow==2.10.0`

---

### **Step 3: Generate Dummy Data** (1 minute)

```bash
# Generate synthetic N2K data for testing
python scripts/00_generate_dummy_data.py
```

**This creates:**
- `data/raw/n2k/normal/dummy_n2k_5000_messages.csv` (fake N2K messages)
- `data/processed/dummy_signals_2000_samples.csv` (fake decoded signals)

**Purpose:** Test the pipeline WITHOUT needing real data!

---

### **Step 4: Test with Dummy Data** (5 minutes)

```bash
# Test initialization phase
python scripts/01_initialize.py

# Check if it worked:
ls results/
# Should see: correlation_matrix.csv, dendrogram.png, etc.

# Test preprocessing phase  
python scripts/02_preprocess_data.py

# Check if it worked:
ls data/processed/views/T_1/
ls data/processed/views/T_5/
# Should see: view_normalized.npy files
```

---

### **Step 5: Fix Any Issues** 

**Common issues at this stage:**

| Issue | Fix |
|-------|-----|
| "No module named tensorflow" | `pip install tensorflow` |
| "File not found" | Check you're in project root |
| "Import error" | Activate venv: `source venv/bin/activate` |
| "N2K decoder fails" | Expected! Decoder is placeholder - uses dummy values for now |

---

## 🎯 **What About Real Data?**

### **You DON'T Need Yet:**
- ❌ Real N2K data from your boats
- ❌ Attack data
- ❌ Large datasets
- ❌ Perfect PGN decoder

### **What This Validates:**
- ✅ Python environment works
- ✅ All modules load correctly
- ✅ Pipeline runs end-to-end
- ✅ No syntax errors or missing dependencies
- ✅ Directory structure is correct

---

## 📊 **When to Collect Real Data:**

**Collect normal N2K data when:**
1. ✅ Validation script passes
2. ✅ Dummy data pipeline works
3. ✅ You've updated the N2K decoder for your PGNs
4. ✅ No errors in initialization/preprocessing

**How much normal data to collect:**
- **Minimum:** 10,000 - 50,000 messages (~30-60 minutes of boat operation)
- **Recommended:** 100,000+ messages (several hours)
- **Best:** Multiple sessions in different conditions

**Collect attack data when:**
- ❌ **NOT YET!** 
- ✅ After training models on normal data
- ✅ After validating detection works
- ✅ Much later in the project

---

## 🔧 **Next Steps After Setup:**

```bash
# 1. Validate everything works
python scripts/00_validate_setup.py

# 2. Generate dummy data
python scripts/00_generate_dummy_data.py

# 3. Test initialization
python scripts/01_initialize.py

# 4. Test preprocessing
python scripts/02_preprocess_data.py

# 5. If all works → Ready to collect real data!
# 6. If errors → Fix issues first
```

---

## ✅ **Setup Checklist:**

```
[ ] Virtual environment created (venv/)
[ ] Dependencies installed (pip install -r requirements.txt)
[ ] Validation script passes (00_validate_setup.py)
[ ] Dummy data generated (00_generate_dummy_data.py)
[ ] Initialization works (01_initialize.py)
[ ] Preprocessing works (02_preprocess_data.py)
[ ] No critical errors
[ ] Ready to collect real N2K data!
```

---

## 💡 **Summary:**

**DO THIS NOW:**
1. Setup environment
2. Install packages
3. Run validation
4. Test with dummy data
5. Fix any bugs

**DO THIS LATER (after setup works):**
1. Update N2K decoder for your PGNs
2. Collect normal N2K data from boats
3. Train models
4. Validate detection
5. **MUCH LATER:** Collect attack data for testing

---

**Current Status: SETUP PHASE - No real data needed yet!** ✅
