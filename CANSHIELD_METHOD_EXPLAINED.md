# 🔍 CANShield Method - Complete Understanding

## 📚 **HOW CANSHIELD REALLY WORKS**

After reading their actual code, here's their EXACT approach:

---

## 1️⃣ **DATA PREPARATION** (`load_dataset.py`)

### Window Creation:
```python
def create_x_sequences(X, time_step, window_step, num_signals, sampling_period):
    X_output = []
    for i in range(0, (len(X) - sampling_period*time_step), window_step):
        X_output.append(X[i : (i + sampling_period*time_step) : sampling_period])
    return np.stack(X_output).reshape(-1, time_step, num_signals, 1)
```

**Key Points:**
- `time_step`: Window length (e.g., 50)
- `window_step`: How much to slide (e.g., 1 for overlapping, 50 for non-overlapping)
- `sampling_period`: Subsampling factor (e.g., 1 = no subsampling)
- **Creates windows incrementally** - NOT all at once!
- Returns shape: `(num_windows, time_step, num_signals, 1)`

---

## 2️⃣ **RECONSTRUCTION & ERROR CALCULATION** (`helper.py`)

### Absolute Error (NOT MSE!):
```python
def calc_loss(x_org, x_recon):
    recon_loss = np.abs(x_org - x_recon)  # ← ABSOLUTE ERROR!
    return recon_loss
```

**Output shape:** Same as input `(num_windows, time_step, num_signals, 1)`

---

## 3️⃣ **TIER 1: SIGNAL-LEVEL THRESHOLDS** (`load_thresholds.py` line 196)

```python
# For EACH signal independently:
for Signal in range(num_signals):
    # Get all errors for this signal across ALL windows and timesteps
    y_train_prob_signal = y_train_prob_org[:, :, Signal].flatten()
    
    # Calculate percentile threshold
    for loss_factor in [95, 99]:  # Can be any percentile
        th = np.percentile(y_train_prob_signal, loss_factor)
        # Store: signal_thresholds[Signal][loss_factor] = th
```

**Result:** 15 independent thresholds (one per signal)

**Example:**
```
Signal 0 (pitch):    threshold_p95 = 0.591
Signal 1 (depth):    threshold_p95 = 0.793
Signal 2 (wind):     threshold_p95 = 0.595
...
```

---

## 4️⃣ **TIER 2: TEMPORAL COUNTING** (`load_thresholds.py` line 244)

```python
# Apply signal thresholds to get binary anomaly indicators
ths_loss_image = [0.591, 0.793, 0.595, ...]  # From Tier 1
y_train_prob_org_bin = (y_train_prob_org > ths_loss_image).astype(int)
# Shape: (num_windows, time_step, num_signals) - binary: 0 or 1

# Count what FRACTION of timesteps are anomalous per signal per window
y_train_prob_org_bin_count = np.sum(y_train_prob_org_bin, axis=1) / time_step
# Shape: (num_windows, num_signals) - values: 0.0 to 1.0

# Then calculate temporal threshold for EACH signal
for Signal in range(num_signals):
    for time_factor in [95, 99]:
        th = np.percentile(y_train_prob_org_bin_count[:, Signal], time_factor)
        # Store: temporal_thresholds[Signal][time_factor] = th
```

**Result:** 15 temporal thresholds (what % of timesteps must be anomalous)

**Example:**
```
Signal 0 (pitch):  If >6% of timesteps are anomalous → signal flagged
Signal 1 (depth):  If >10% of timesteps are anomalous → signal flagged
...
```

---

## 5️⃣ **TIER 3: MULTI-SIGNAL VOTING** (`load_thresholds.py` line 288)

```python
# Apply temporal thresholds to get signal-level binary flags
ths_time_image = [0.06, 0.10, 0.08, ...]  # From Tier 2
y_train_prob_org_sig_count = (y_train_prob_org_bin_count > ths_time_image).astype(int)
# Shape: (num_windows, num_signals) - binary: 0 or 1

# Count what FRACTION of signals are flagged per window
y_train_prob_org_sig_count = np.sum(y_train_prob_org_sig_count, axis=1) / num_signals
# Shape: (num_windows,) - values: 0.0 to 1.0

# Calculate final threshold
for signal_factor in [95, 99]:
    th = np.percentile(y_train_prob_org_sig_count, signal_factor)
    # Store: final_threshold[signal_factor] = th
```

**Result:** ONE final threshold (what % of signals must flag for attack)

**Example:**
```
If >6.7% of signals (i.e., 1 out of 15) flag the window → ATTACK!
```

---

## 6️⃣ **TESTING/PREDICTION** (`load_predictions.py`)

```python
# Phase 1: Calculate reconstruction errors
y_test_prob_org, y_test_seq = generate_and_save_prediction_loss_per_file(...)
# Shape: (num_windows, time_step, num_signals, 1)

# Phase 2: Apply Tier 1 thresholds
loss_df = pd.read_csv('thresholds_loss_...')
ths_loss_image = loss_df[loss_df['loss_factor'] == 95]['th'].values
y_test_prob_org_bin = (y_test_prob_org > ths_loss_image).astype(int)
y_test_prob_org_bin_count = np.sum(y_test_prob_org_bin, 1) / time_step

# Phase 3: Apply Tier 2 thresholds
time_df = pd.read_csv('thresholds_time_...')
ths_time_image = time_df[time_df['time_factor'] == 95]['th'].values
y_test_prob_org_sig_count = (y_test_prob_org_bin_count > ths_time_image).astype(int)
y_test_prob_org_sig_count = np.sum(y_test_prob_org_sig_count, 1) / num_signals

# Phase 4: Apply Tier 3 threshold
signal_df = pd.read_csv('thresholds_signal_...')
final_th = signal_df[signal_df['signal_factor'] == 95]['th'].values[0]
predictions = (y_test_prob_org_sig_count > final_th).astype(int)
```

---

## 🔑 **KEY DIFFERENCES FROM OUR APPROACH**

### ❌ What We Did Wrong:

**1. Memory Issue:**
```python
# OUR APPROACH - Tried to store everything!
all_signal_errors = np.zeros((4.9M, 50, 15))  # 3.7 GB!
for batch in batches:
    errors = ...
    all_signal_errors[i:i+batch_size] = errors  # ← Memory overflow!
```

**2. Processing Flow:**
- We tried to calculate ALL errors first, THEN apply thresholds
- CANShield calculates errors PER FILE, applies thresholds immediately

### ✅ What CANShield Does Right:

**1. File-by-File Processing:**
```python
# Their approach - Process one file at a time
for file_name, file_path in file_dict.items():
    # Load ONE file
    x_seq, y_seq = load_data_create_images(...)  # Smaller dataset
    
    # Reconstruct
    x_recon = autoencoder.predict(x_seq)
    
    # Calculate errors
    y_prob_org = calc_loss(x_seq, x_recon)
    
    # Use errors to calculate/update thresholds
    # Then DISCARD x_seq, x_recon, y_prob_org
```

**2. Dictionary Caching:**
```python
# They cache predictions in a dictionary
y_train_prob_org_dict[f"{file_name}_{time_step}_{sampling_period}"] = y_prob_org

# Reuse if already calculated
try:
    y_prob_org = y_train_prob_org_dict[key]
except:
    y_prob_org = calculate_new(...)
```

**3. Incremental Threshold Building:**
- Don't need ALL data at once
- Process file 1 → add to threshold dataframe
- Process file 2 → add to threshold dataframe
- ...
- At the end: `groupby().mean()` to get final thresholds

---

## 🎯 **WHAT WE NEED TO IMPLEMENT**

### For OUR Maritime Dataset:

**We have ONE BIG file** (attack_dataset.npz with 4.9M timesteps), not multiple small files.

**Solution:**
1. **Split our dataset into chunks** (e.g., 100K timesteps per chunk)
2. **Process each chunk like CANShield processes files**
3. **Build thresholds incrementally**

### Pseudocode:
```python
# Split dataset
CHUNK_SIZE = 100000  # timesteps per chunk
chunks = split_into_chunks(attack_data, CHUNK_SIZE)

# Phase 1: Calculate Tier 1 thresholds (per-signal)
signal_errors_dict = {sig: [] for sig in range(15)}

for chunk in normal_chunks:
    # Create windows from this chunk
    windows = create_sliding_windows(chunk)  # Smaller!
    
    # Reconstruct
    reconstructed = model.predict(windows)
    
    # Calculate absolute errors
    abs_errors = np.abs(windows - reconstructed)  # (N, 50, 15, 1)
    
    # Store per-signal errors
    for sig in range(15):
        signal_errors_dict[sig].append(abs_errors[:, :, sig].flatten())
    
    # DISCARD windows, reconstructed, abs_errors (free memory!)

# Calculate thresholds
signal_thresholds = {}
for sig in range(15):
    all_errors = np.concatenate(signal_errors_dict[sig])
    signal_thresholds[sig] = np.percentile(all_errors, 95)

# Phase 2 & 3: Similar chunked processing...
```

---

## 📊 **SUMMARY**

**CANShield's Brilliance:**
1. ✅ Process data in SMALL chunks (file-by-file)
2. ✅ Use absolute error (more sensitive than MSE)
3. ✅ 3-tier hierarchical detection (per-signal → temporal → multi-signal)
4. ✅ Incremental threshold calculation (don't need all data at once)
5. ✅ Cache intermediate results in dictionaries

**Our Implementation Plan:**
1. Split 4.9M timesteps into manageable chunks (100K each)
2. Process each chunk separately
3. Build thresholds incrementally
4. Apply 3-tier detection exactly like CANShield
5. Should work with our memory constraints!

---

**Ready to implement?** This is the correct approach that will solve our memory problem and give us proper hierarchical detection!
