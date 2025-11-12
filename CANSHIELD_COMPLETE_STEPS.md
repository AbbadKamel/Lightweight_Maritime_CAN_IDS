# CANShield - Complete Implementation Steps

**Reference:** CANShield: Deep-Learning-Based Intrusion Detection Framework for Controller Area Networks at the Signal Level (IEEE IoT Journal, 2023)

**Project:** Lightweight_IA_V_2  
**Date:** November 3, 2025

---

## 📚 **OVERVIEW**

CANShield is a signal-level intrusion detection system for CAN bus networks using:
- **Multi-scale temporal analysis** (sampling periods: 1, 5, 10, 20, 50)
- **CNN autoencoders** for anomaly detection
- **Three-tier thresholding** for progressive filtering
- **Ensemble decision** across multiple models

**Detection Latency:** <10ms (real-time capable)

---

# 🔧 **INITIALIZATION PHASE** (One-Time Setup)

**Purpose:** Prepare signal set and ordering before any training

---

## **Step 1: Select Critical Signals**
**What:** Choose m signals (e.g., m=20 for SynCAN dataset)  
**How:** 
- Use domain knowledge to identify high-priority signals
- Examples: Engine RPM, Vehicle Speed, Steering Angle, Brake Pressure, etc.

**Why:** Focus on most security-critical signals

---

## **Step 2: Add Correlated Signals**
**What:** For each critical signal, add highly correlated signals  
**How:**
- Compute correlation between all available signals
- Add signals with high correlation to critical ones

**Why:** Improves robustness - attacks often affect correlated signals together

---

## **Step 3: Compute Pearson Correlation Matrix**
**What:** Calculate pairwise correlation between all m selected signals  
**Formula:** 
```
r(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
```

**Output:** m × m correlation matrix

---

## **Step 4: Run Hierarchical Clustering**
**What:** Cluster signals based on correlation matrix  
**Method:**
- Use hierarchical clustering (linkage method: complete/average)
- Build dendrogram showing signal relationships

**Why:** Groups similar signals together

---

## **Step 5: Reorder Signals by Correlation**
**What:** Arrange signals so highly-correlated ones are adjacent  
**How:**
- Use dendrogram to determine optimal ordering
- Signals with high correlation placed next to each other

**Why:** 
- Small CNN filters (3×3) can learn inter-signal relationships
- Adjacent correlated signals help convolution detect patterns

**Important:** This ordering is **FIXED** for all training and deployment

---

## **Initialization Output:**
✅ m signals selected (e.g., 20)  
✅ Signals ordered by correlation  
✅ Order saved and used for all subsequent phases

---

# 🔷 **PHASE 1: DATA PREPROCESSING MODULE**

**Purpose:** Transform raw CAN messages → Multi-scale 2D views for CNN

---

## **Step 1: Create FIFO Queue Q**
**What:** Maintain rolling window of last q time steps  
**Parameters:**
- q = 1000 time steps (configurable)
- FIFO: First In, First Out

**Structure:**
```
Q = [t_current-999, t_current-998, ..., t_current-1, t_current]
    Shape: q × m (1000 × 20)
```

---

## **Step 2: Decode Incoming CAN Messages**
**What:** Extract signal values from raw CAN payloads  
**Methods:**
- **Option 1:** Use vehicle's DBC file (if available)
- **Option 2:** Use CAN-D reverse engineering (if DBC unavailable)

**Process:**
```
CAN Message → DBC/Mapping → Signal Values
(ID, Payload) →            → (RPM=2500, Speed=80, ...)
```

---

## **Step 3: Update Queue Q**
**What:** Add new time step to Q  
**Process:**
1. **Update only signals present** in the new CAN message
2. For signals **not updated** this tick: **Forward-fill** from last known value
3. Add new time step to end of Q
4. Remove oldest time step (maintain size q)

**Example:**
```
New CAN message updates: RPM, Speed (2 signals)
Other 18 signals: Keep previous values (forward-fill)
```

**Why forward-fill:** CAN signals don't update synchronously; assume unchanged until next update

---

## **Step 4: Build Multiple Views (Multi-Scale)**
**What:** Create n different views from same queue Q  
**Sampling Periods:** T_x ∈ [1, 5, 10, 20, 50]

**For each T_x:**
1. Sample every T_x-th time step from Q
2. **Always take first w columns** (e.g., w=50)
3. Create view D_x of shape: **m × w** (20 × 50)

**Example:**
```
T_1 = 1:  D_1 = Q[0:50:1]   → [t_0, t_1, t_2, ..., t_49]
T_5 = 5:  D_5 = Q[0:250:5]  → [t_0, t_5, t_10, ..., t_245]
T_10 = 10: D_10 = Q[0:500:10] → [t_0, t_10, t_20, ..., t_490]
```

**All views:** Same shape m × w (20 × 50)

---

## **Step 5: Purpose of Multi-View**
**Why different sampling periods?**

| View | Sampling Period | Captures | Best For |
|------|----------------|----------|----------|
| D_1  | T_1 = 1        | Short/fast patterns | Flooding, Fast attacks |
| D_5  | T_5 = 5        | Medium patterns | Plateau, Playback |
| D_10 | T_10 = 10      | Long/slow trends | Suppress, Continuous |
| D_20 | T_20 = 20      | Very slow trends | Gradual drift |
| D_50 | T_50 = 50      | Ultra-slow trends | Long-term attacks |

**Benefit:** Ensemble captures attacks at **all time scales**

---

## **Step 6: Why Fixed w Across All Views?**
**What:** Keep w=50 constant for all views  
**Why:**
- Allows **reusing same CNN architecture** for all AEs
- Simplifies implementation
- All autoencoders have identical structure

---

## **Step 7: Normalize Signals**
**What:** Apply Min-Max scaling to [0, 1]  
**Formula:**
```
X_normalized = (X - X_min) / (X_max - X_min)
```

**How:**
- Use pre-computed min/max from training data
- Apply same scaling in training and deployment

**Why:** 
- Neural networks work better with normalized inputs
- All signals on same scale

---

## **Step 8: Reshape for CNN**
**What:** Add channel dimension  
**Transform:**
```
m × w  →  m × w × 1
20 × 50 → 20 × 50 × 1
```

**Why:** CNN expects (height, width, channels) format

---

## **Phase 1 Output:**
✅ Multiple 2D views: D_1, D_5, D_10, D_20, D_50  
✅ All views: Shape m × w × 1 (20 × 50 × 1)  
✅ Normalized to [0, 1]  
✅ Ready for CNN autoencoders

---

# 🔷 **PHASE 2: DATA ANALYZER MODULE (TRAINING)**

**Purpose:** Train CNN autoencoders to learn normal patterns

---

## **ARCHITECTURE DEFINITION**

### **Step 1: CNN Autoencoder Structure**

**Input Shape:** m × w × 1 (20 × 50 × 1)

**ENCODER (Compression):**
```
Layer 1: Conv2D(32 filters, 3×3, padding='same') + LeakyReLU(α=0.2)
         → MaxPooling2D(2×2)
         
Layer 2: Conv2D(16 filters, 3×3, padding='same') + LeakyReLU(α=0.2)
         → MaxPooling2D(2×2)
         
Layer 3: Conv2D(16 filters, 3×3, padding='same') + LeakyReLU(α=0.2)
         → MaxPooling2D(2×2)
```

**BOTTLENECK:**
- Compressed latent representation
- Encodes compact "state" of CAN network

**DECODER (Reconstruction):**
```
Layer 4: Conv2D(16 filters, 3×3, padding='same') + LeakyReLU(α=0.2)
         → UpSampling2D(2×2)
         
Layer 5: Conv2D(32 filters, 3×3, padding='same') + LeakyReLU(α=0.2)
         → UpSampling2D(2×2)
         
Layer 6: Conv2D(1 filter, 3×3, padding='same') + Sigmoid
         → Cropping (to match input size)
```

**Output Shape:** m × w × 1 (20 × 50 × 1) - Same as input

**Total:** 5 convolutional layers  
**Filter Progression:** 32 → 16 → 16 → 32 → 1

---

### **Step 2: Why This Architecture?**

**3×3 Kernels:** 
- Small receptive field
- Captures local signal patterns
- Learns inter-signal relationships (thanks to correlation ordering)

**LeakyReLU(α=0.2):**
- Prevents dying ReLU problem
- Allows small gradients for negative values

**Sigmoid Output:**
- Output range [0, 1]
- Matches normalized input range

**No Explicit States:**
- Bottleneck automatically encodes compact representation
- No need to manually define vehicle states

---

## **TRAINING SETUP**

### **Step 3: Configure Training Parameters**

**EXACT HYPERPARAMETERS (Copy Verbatim):**
```python
Optimizer: Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999)
Loss: MSE (Mean Squared Error)
Batch Size: 128
Epochs: 100
Validation Split: 10% (hold-out from training data)
```

**Compilation:**
```python
autoencoder.compile(
    optimizer=Adam(lr=0.0002),
    loss='mse',
    metrics=['mae']
)
```

---

### **Step 4: Prepare Training Data**

**Data Source:** Normal driving data ONLY (no attacks)

**Examples:**
- SynCAN: train_1.csv, train_2.csv, train_3.csv, train_4.csv
- ROAD: ambient_*.csv

**Process:**
1. Load normal CAN data
2. Apply Phase 1 preprocessing → Get views D_x
3. Split: 90% training, 10% validation

---

## **MULTI-VIEW TRAINING WITH TRANSFER LEARNING**

### **Step 5: Train First Autoencoder (T_1 = 1)**

**What:** Train AE_1 from scratch  
**Data:** View D_1 (sampling_period = 1)  
**Process:**
```python
history_1 = AE_1.fit(
    D_1_train,           # Input
    D_1_train,           # Target (SAME!)
    batch_size=128,
    epochs=100,
    validation_data=(D_1_val, D_1_val),
    callbacks=[EarlyStopping(patience=10)]
)
```

**Goal:** Learn to reconstruct D_1 ≈ D̂_1 with minimal MSE

**Save:** AE_1 model

---

### **Step 6: Train Subsequent Autoencoders (Transfer Learning)**

**For T_x in [5, 10, 20, 50]:**

**Transfer Learning Process:**
```python
# 1. Load previous autoencoder
AE_x = load_model(f'AE_{previous_T}.h5')

# 2. Initialize weights from previous model
# (Already loaded - weights transferred)

# 3. Fine-tune on new sampling period
history_x = AE_x.fit(
    D_x_train,           # New view
    D_x_train,
    batch_size=128,
    epochs=100,          # Can use fewer epochs for fine-tuning
    validation_data=(D_x_val, D_x_val)
)

# 4. Save fine-tuned model
AE_x.save(f'AE_{T_x}.h5')
```

**Why Transfer Learning?**
- Previous AE already learned basic CAN patterns
- Fine-tuning is faster and often better than training from scratch
- Reduces computational cost

**Example Progression:**
```
AE_1 (T=1):  Train from scratch (100 epochs)
AE_5 (T=5):  Load AE_1 → Fine-tune (50 epochs)
AE_10 (T=10): Load AE_5 → Fine-tune (50 epochs)
AE_20 (T=20): Load AE_10 → Fine-tune (50 epochs)
AE_50 (T=50): Load AE_20 → Fine-tune (50 epochs)
```

---

### **Step 7: Validation During Training**

**What:** Monitor performance on 10% hold-out validation set  
**Metrics:**
- Training loss (MSE)
- Validation loss (MSE)
- Mean Absolute Error (MAE)

**Early Stopping:**
- If validation loss doesn't improve for 10 epochs → Stop
- Prevents overfitting

**Save Best Model:**
- Keep model with lowest validation loss

---

## **THRESHOLD GENERATION (After Training)**

**Purpose:** Determine detection thresholds using normal data

---

### **Step 8: Prepare Threshold Data**

**Options:**
- **Option 1:** Use entire training set
- **Option 2:** Use 10% random hold-out from training

**Recommended:** Use hold-out to avoid overfitting thresholds

---

### **Step 9: Run AEs on Threshold Data**

**For each AE_x:**
1. Pass threshold data through autoencoder
2. Get reconstructed output D̂_x
3. Calculate reconstruction loss: **L_x = |D_x - D̂_x|**
4. Shape of L_x: m × w (20 × 50) per sample

---

### **Step 10: Grid Search for Optimal Thresholds**

**Search Space:** Percentiles p, q, r ∈ [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.9, 99.99]

**For each AE_x, compute three thresholds:**

---

#### **Tier 1: R_Loss_x (Per-Signal Loss Threshold)**

**What:** Threshold for pixel-level anomaly detection  
**How:**
```python
for signal_i in range(m):  # For each of 20 signals
    loss_values_i = L_x[:, signal_i, :]  # All samples, signal i, all time
    R_Loss_x[signal_i] = np.percentile(loss_values_i, p)
```

**Example with p=95:**
```
Signal 0: R_Loss_1[0] = 95th percentile of loss for signal 0 = 0.15
Signal 1: R_Loss_1[1] = 95th percentile of loss for signal 1 = 0.12
...
Signal 19: R_Loss_1[19] = 95th percentile of loss for signal 19 = 0.18
```

**Output:** Vector R_Loss_x of length m (20 thresholds)

---

#### **Tier 2: R_Time_x (Per-Signal Time-Violation Threshold)**

**What:** Threshold for temporal anomaly detection  
**How:**
```python
# 1. Apply R_Loss to get binary anomaly matrix
B_x = (L_x > R_Loss_x).astype(int)  # Shape: (samples, m, w)

# 2. Count violations per signal across time
V_x = np.sum(B_x, axis=2)  # Sum over w (time dimension)
                           # Shape: (samples, m)

# 3. For each signal, compute percentile threshold
for signal_i in range(m):
    R_Time_x[signal_i] = np.percentile(V_x[:, signal_i], q)
```

**Example with q=99:**
```
Signal 0: R_Time_1[0] = 99th percentile of violation counts = 8
Signal 1: R_Time_1[1] = 99th percentile of violation counts = 6
...
```

**Meaning:** If signal has >8 violated pixels in a window, it's suspicious

**Output:** Vector R_Time_x of length m (20 thresholds)

---

#### **Tier 3: R_Signal_x (Overall Signal-Violation Threshold)**

**What:** Threshold for window-level attack detection  
**How:**
```python
# 1. Apply R_Time to get suspicious signals
S_x = (V_x > R_Time_x).astype(int)  # Shape: (samples, m)

# 2. Count suspicious signals per window
S_count = np.sum(S_x, axis=1)  # Sum over m (signals)
                               # Shape: (samples,)

# 3. Compute percentile threshold
R_Signal_x = np.percentile(S_count, r)
```

**Example with r=95:**
```
R_Signal_1 = 95th percentile of suspicious signal counts = 3
```

**Meaning:** If >3 signals are suspicious in a window, it's an attack

**Output:** Scalar R_Signal_x (1 threshold per AE)

---

### **Step 11: Compute Ensemble Threshold**

**What:** Average threshold across all autoencoders  
**Formula:**
```python
R_Signal_ens = mean([R_Signal_1, R_Signal_5, R_Signal_10, R_Signal_20, R_Signal_50])
```

**Example:**
```
R_Signal_1 = 3
R_Signal_5 = 4
R_Signal_10 = 5
R_Signal_20 = 4
R_Signal_50 = 6

R_Signal_ens = (3 + 4 + 5 + 4 + 6) / 5 = 4.4
```

**Used for:** Final ensemble attack decision

---

### **Step 12: Save All Thresholds**

**File Structure:**
```
thresholds/
├── AE_1_R_Loss.csv      (20 values)
├── AE_1_R_Time.csv      (20 values)
├── AE_1_R_Signal.csv    (1 value)
├── AE_5_R_Loss.csv
├── AE_5_R_Time.csv
├── AE_5_R_Signal.csv
├── ...
└── R_Signal_ens.csv     (1 value)
```

---

### **Step 13: Save Trained Models**

**File Structure:**
```
models/
├── AE_1.h5
├── AE_5.h5
├── AE_10.h5
├── AE_20.h5
└── AE_50.h5
```

---

## **Phase 2 Output:**
✅ 5 trained CNN autoencoders (AE_1, AE_5, AE_10, AE_20, AE_50)  
✅ Tier 1 thresholds: R_Loss_x for each AE (20 values each)  
✅ Tier 2 thresholds: R_Time_x for each AE (20 values each)  
✅ Tier 3 thresholds: R_Signal_x for each AE (1 value each)  
✅ Ensemble threshold: R_Signal_ens (1 value)

---

# 🔷 **PHASE 3: ATTACK DETECTION MODULE (DEPLOYMENT)**

**Purpose:** Real-time attack detection using trained models

---

## **REAL-TIME PROCESSING LOOP**

### **Step 1: Initialize System**

**Load at startup:**
```python
# Load all trained autoencoders
AE_1 = load_model('AE_1.h5')
AE_5 = load_model('AE_5.h5')
AE_10 = load_model('AE_10.h5')
AE_20 = load_model('AE_20.h5')
AE_50 = load_model('AE_50.h5')

# Load all thresholds
R_Loss_1 = load_csv('AE_1_R_Loss.csv')
R_Time_1 = load_csv('AE_1_R_Time.csv')
R_Signal_1 = load_csv('AE_1_R_Signal.csv')
# ... (repeat for all AEs)

R_Signal_ens = load_csv('R_Signal_ens.csv')

# Initialize queue Q
Q = initialize_queue(size=1000, num_signals=20)
```

---

### **Step 2: For Each New Time Step (Continuous Loop)**

**Process new CAN message:**
```python
while True:
    # Receive CAN message
    can_msg = receive_can_message()
    
    # Decode using DBC
    signal_values = decode_can_message(can_msg, dbc_file)
    
    # Update queue Q (Step 3)
    Q = update_queue(Q, signal_values)
    
    # Rebuild views (Step 4)
    views = build_views(Q)
    
    # Run detection (Steps 5-10)
    decision = detect_attack(views, AEs, thresholds)
    
    # Take action (Step 11)
    if decision == ATTACK:
        raise_alarm()
```

---

### **Step 3: Update Queue Q**

**What:** Add new time step to queue  
**Process:**
```python
def update_queue(Q, signal_values):
    # Create new row
    new_row = [None] * m  # m = 20 signals
    
    # Update only signals present in this message
    for signal_id, value in signal_values.items():
        signal_index = get_signal_index(signal_id)
        new_row[signal_index] = value
    
    # Forward-fill missing signals from last row
    last_row = Q[-1]
    for i in range(m):
        if new_row[i] is None:
            new_row[i] = last_row[i]
    
    # Add to queue, remove oldest
    Q = np.vstack([Q[1:], new_row])  # Shift up, add new
    
    return Q
```

**Queue always maintains:** Last 1000 time steps

---

### **Step 4: Rebuild All Views from Q**

**What:** Create D_1, D_5, D_10, D_20, D_50 from updated Q  
**Process:**
```python
def build_views(Q):
    views = {}
    
    for T_x in [1, 5, 10, 20, 50]:
        # Sample every T_x-th step, take first w
        sampled = Q[::T_x][:w]  # w = 50
        
        # Normalize
        normalized = normalize(sampled)
        
        # Reshape to m × w × 1
        view = normalized.reshape(m, w, 1)
        
        views[T_x] = view
    
    return views
```

**Output:** 5 views, each shape (20, 50, 1)

---

## **THREE-STEP ANALYSIS (Algorithm 2)**

### **Step 5: Run All Autoencoders**

**For each view:**
```python
reconstructions = {}
losses = {}

for T_x in [1, 5, 10, 20, 50]:
    # Get corresponding autoencoder
    AE_x = AEs[T_x]
    
    # Reconstruct
    D_x = views[T_x]
    D_hat_x = AE_x.predict(D_x)
    
    # Calculate reconstruction loss
    L_x = np.abs(D_x - D_hat_x)
    
    reconstructions[T_x] = D_hat_x
    losses[T_x] = L_x
```

**L_x shape:** (20, 50, 1) → Reconstruction loss per pixel

---

### **Step 6: Step 1 - Pixel-Level Detection (Tier 1)**

**Apply loss thresholds:**
```python
def tier1_detection(L_x, R_Loss_x):
    # L_x: (m, w, 1) - reconstruction loss
    # R_Loss_x: (m,) - threshold per signal
    
    B = np.zeros((m, w))
    
    for i in range(m):  # For each signal
        for j in range(w):  # For each time step
            if L_x[i, j, 0] > R_Loss_x[i]:
                B[i, j] = 1  # Violation
            else:
                B[i, j] = 0  # Normal
    
    return B
```

**Output:** Binary matrix B (20 × 50)
- B[i,j] = 1 → Pixel (signal i, time j) is anomalous
- B[i,j] = 0 → Pixel is normal

---

### **Step 7: Step 2 - Signal-Level Detection (Tier 2)**

**Count violations and apply time thresholds:**
```python
def tier2_detection(B, R_Time_x):
    # B: (m, w) - binary anomaly matrix
    # R_Time_x: (m,) - threshold per signal
    
    V = np.zeros(m)  # Violation counts
    S = np.zeros(m)  # Suspicious signals
    
    for i in range(m):  # For each signal
        # Count violations across time
        V[i] = np.sum(B[i, :])
        
        # Compare with threshold
        if V[i] > R_Time_x[i]:
            S[i] = 1  # Signal is suspicious
        else:
            S[i] = 0  # Signal is normal
    
    return V, S
```

**Output:** 
- V (20,) - Violation counts per signal
- S (20,) - Binary vector of suspicious signals

---

### **Step 8: Step 3 - Window-Level Anomaly Score**

**Calculate anomaly score for this AE:**
```python
def tier3_score(S):
    # S: (m,) - binary suspicious signals vector
    
    # Count suspicious signals
    num_suspicious = np.sum(S)
    
    # Calculate anomaly score (proportion)
    P_x = num_suspicious / m
    
    return P_x
```

**Output:** P_x ∈ [0, 1]
- P_x = 0 → No suspicious signals (normal)
- P_x = 1 → All signals suspicious (definite attack)
- P_x = 0.3 → 30% of signals suspicious

---

### **Step 9: Repeat for All Autoencoders**

**Run 3-step analysis for each AE:**
```python
anomaly_scores = {}

for T_x in [1, 5, 10, 20, 50]:
    # Get loss matrix
    L_x = losses[T_x]
    
    # Step 1: Tier 1 detection
    B_x = tier1_detection(L_x, R_Loss[T_x])
    
    # Step 2: Tier 2 detection
    V_x, S_x = tier2_detection(B_x, R_Time[T_x])
    
    # Step 3: Tier 3 score
    P_x = tier3_score(S_x)
    
    anomaly_scores[T_x] = P_x
```

**Output:** 5 anomaly scores (P_1, P_5, P_10, P_20, P_50)

---

## **ENSEMBLE DECISION**

### **Step 10: Combine Scores from All AEs**

**Average ensemble:**
```python
def ensemble_decision(anomaly_scores, R_Signal_ens):
    # anomaly_scores: {1: P_1, 5: P_5, 10: P_10, 20: P_20, 50: P_50}
    
    # Average across all AEs
    P_ens = np.mean(list(anomaly_scores.values()))
    
    # Compare with ensemble threshold
    if P_ens > R_Signal_ens:
        decision = ATTACK
    else:
        decision = BENIGN
    
    return decision, P_ens
```

**Example:**
```
P_1 = 0.15
P_5 = 0.20
P_10 = 0.25
P_20 = 0.10
P_50 = 0.30

P_ens = (0.15 + 0.20 + 0.25 + 0.10 + 0.30) / 5 = 0.20

R_Signal_ens = 0.15

Decision: 0.20 > 0.15 → ATTACK DETECTED! 🚨
```

---

### **Step 11: Take Action**

**If ATTACK:**
```python
if decision == ATTACK:
    # Log attack
    log_attack(timestamp, P_ens, anomaly_scores)
    
    # Raise alarm
    raise_alarm()
    
    # Optional: Take defensive action
    # - Alert driver
    # - Enter safe mode
    # - Isolate affected ECU
```

**If BENIGN:**
```python
else:
    # Continue normal operation
    continue_monitoring()
```

---

## **WHY ENSEMBLE WORKS**

### **Different Temporal Scales Capture Different Attacks:**

| Attack Type | Speed | Best Detected By |
|------------|-------|------------------|
| Flooding | Very fast | AE_1 (T=1) |
| Playback | Fast-medium | AE_1, AE_5 |
| Plateau | Medium | AE_5, AE_10 |
| Continuous | Slow | AE_10, AE_20 |
| Suppress | Very slow | AE_20, AE_50 |

**Averaging Benefits:**
- **Reduces false positives:** Single AE might spike, average smooths
- **Captures multi-speed attacks:** Some attacks affect multiple time scales
- **Robust detection:** Attack must fool ALL models to evade

**Trade-offs:**
- Slightly higher latency for very slow attacks
- May miss attacks that only affect one time scale
- Overall: Better precision/recall trade-off

---

## **PERFORMANCE METRICS**

### **Detection Latency:**
- **Per AE inference:** ~2ms
- **5 AEs total:** ~10ms
- **Total (with preprocessing):** <10ms ✅

**Real-time capable for CAN bus (typical: 10-100ms between messages)**

---

## **Phase 3 Output:**
✅ Real-time attack/benign decisions  
✅ Anomaly scores per autoencoder  
✅ Ensemble anomaly score  
✅ Detection latency <10ms  
✅ Attack logs and alarms

---

# 📝 **CRITICAL IMPLEMENTATION NOTES**

## **1. Input Format**
- **2D views:** m × w (20 × 50) per sampling period
- **With channel:** m × w × 1 (20 × 50 × 1)
- **Same architecture** for all AEs → Code reuse

---

## **2. Autoencoder Rationale**
- **No explicit vehicle states needed**
- **Bottleneck:** Automatically learns compact representation
- **Anomalies:** Higher reconstruction loss (can't reconstruct abnormal patterns)

---

## **3. Exact Hyperparameters (Copy Verbatim)**

```python
# Architecture
conv_filters = [32, 16, 16, 32, 1]
kernel_size = (3, 3)
activation = LeakyReLU(alpha=0.2)
output_activation = 'sigmoid'

# Training
optimizer = Adam(learning_rate=0.0002)
loss = 'mse'
batch_size = 128
epochs = 100

# Normalization
scaling = MinMaxScaler(feature_range=(0, 1))
```

**Why verbatim?** Paper tested these specific values - deviation may reduce performance

---

## **4. Key Design Choices**

### **Correlation-Based Clustering:**
- Groups similar signals together
- Adjacent signals are correlated
- **Helps CNN filters** learn inter-signal patterns

### **Fixed w Across Views:**
- w = 50 for all sampling periods
- Allows identical architecture
- Simplifies deployment

### **Transfer Learning:**
- Train T_1 thoroughly
- Fine-tune for higher T_x
- **Reduces training time** by 50-70%

### **Three-Tier Thresholding:**
- Progressive filtering
- Each tier reduces false positives
- **Final threshold** highly selective

### **Ensemble Averaging:**
- Combines all temporal scales
- More robust than single model
- **Reduces variance** in decisions

---

# 🎯 **SUMMARY CHECKLIST**

## **Initialization:**
- [ ] Select m critical signals (domain knowledge)
- [ ] Add correlated signals for robustness
- [ ] Compute Pearson correlation matrix
- [ ] Run hierarchical clustering
- [ ] Reorder signals by correlation
- [ ] Save fixed signal ordering

## **Phase 1 - Preprocessing:**
- [ ] Create FIFO queue Q (size q)
- [ ] Decode CAN messages (DBC/CAN-D)
- [ ] Update queue (forward-fill missing)
- [ ] Build multi-scale views D_x for T_x ∈ [1,5,10,20,50]
- [ ] Keep w=50 fixed across views
- [ ] Normalize to [0, 1]
- [ ] Reshape to m×w×1

## **Phase 2 - Training:**
- [ ] Define CNN autoencoder (5 conv layers: 32→16→16→32→1)
- [ ] Use 3×3 kernels, LeakyReLU(0.2), sigmoid output
- [ ] Configure: Adam(lr=0.0002), MSE, batch=128, epochs=100
- [ ] Train AE_1 from scratch on normal data
- [ ] Transfer learning for AE_5, AE_10, AE_20, AE_50
- [ ] Grid search thresholds p,q,r ∈ [90-99.99]
- [ ] Compute R_Loss_x (Tier 1) for each AE
- [ ] Compute R_Time_x (Tier 2) for each AE
- [ ] Compute R_Signal_x (Tier 3) for each AE
- [ ] Compute R_Signal_ens = average(R_Signal_x)
- [ ] Save all models and thresholds

## **Phase 3 - Detection:**
- [ ] Load all AEs and thresholds at startup
- [ ] Initialize queue Q
- [ ] For each new CAN message:
  - [ ] Decode and update Q
  - [ ] Build views D_x
  - [ ] Run all AEs → get L_x
  - [ ] Tier 1: Apply R_Loss → B
  - [ ] Tier 2: Apply R_Time → S
  - [ ] Tier 3: Calculate P_x
  - [ ] Ensemble: P_ens = mean(P_x)
  - [ ] Decision: P_ens > R_Signal_ens?
  - [ ] If attack: Raise alarm
- [ ] Maintain <10ms latency

---

# 📊 **EXPECTED RESULTS (SynCAN Dataset)**

| Model | Flooding | Suppress | Plateau | Continuous | Playback | Average |
|-------|----------|----------|---------|------------|----------|---------|
| AE_1 | 0.940 | 0.860 | 0.907 | 0.853 | 0.927 | 0.898 |
| AE_5 | 0.997 | 0.976 | 0.944 | 0.837 | 0.905 | 0.932 |
| AE_10 | 0.994 | 0.978 | 0.924 | 0.814 | 0.888 | 0.920 |
| **Ensemble** | **0.997** | **0.985** | **0.961** | **0.870** | **0.948** | **0.952** |

**Metric:** AUPRC (Area Under Precision-Recall Curve)

---

# 🚀 **NEXT STEPS FOR IMPLEMENTATION**

1. **Setup Environment:** Install TensorFlow, Keras, NumPy, Pandas, scikit-learn
2. **Prepare Data:** Obtain CAN dataset (SynCAN or collect own)
3. **Implement Initialization:** Signal selection and correlation clustering
4. **Implement Phase 1:** Data preprocessing module
5. **Implement Phase 2:** Train autoencoders with transfer learning
6. **Implement Phase 3:** Real-time detection module
7. **Test & Validate:** Compare results with paper
8. **Optimize:** Reduce latency, improve accuracy
9. **Deploy:** Integrate with real CAN bus system

---

**Document Version:** 1.0  
**Last Updated:** November 3, 2025  
**Status:** Ready for Implementation  

---

End of Document.
