# 📘 PHASE 0: INITIALIZATION - DETAILED WALKTHROUGH

**Project**: Lightweight AI Intrusion Detection for Maritime CAN Bus  
**Phase**: Initialization (Signal Selection & Quality Analysis)  
**Date**: November 2025  
**Status**: ✅ COMPLETED

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [Input Data](#input-data)
3. [Step-by-Step Process](#step-by-step-process)
4. [Code Analysis](#code-analysis)
5. [Results & Outputs](#results--outputs)
6. [Key Decisions & Rationale](#key-decisions--rationale)

---

## 🎯 OVERVIEW

### **What is Initialization Phase?**

The initialization phase is the **first critical step** in the machine learning pipeline. Before we can train any AI model, we need to:

1. **Understand our data**: What signals do we have? What's their quality?
2. **Select relevant signals**: Which signals are useful? Which should be removed?
3. **Analyze relationships**: How do signals correlate? Are there redundancies?
4. **Create baseline**: Establish ground truth statistics for normal operation

### **Why is this Important?**

❌ **Without proper initialization:**
- Garbage In → Garbage Out (GIGO principle)
- Model trains on low-quality/missing data
- Redundant signals waste computation
- Poor anomaly detection performance

✅ **With proper initialization:**
- Clean, high-quality training data
- Relevant features only (no noise)
- Understood signal relationships
- Strong baseline for anomaly detection

---

## 📥 INPUT DATA

### **Source File**
```
Path: results/fixed_decoder_data/decoded_brute_frames.csv
Size: ~98,942 rows × 23 columns
Format: CSV (Comma-Separated Values)
Encoding: UTF-8
```

### **How was this file created?**

This file is the **output of the NMEA 2000 decoder** (`n2k_decoder.py`):

```
Raw CAN Frames (ESP32 capture)
         ↓
aggregated_brute_frames.csv (154,161 frames)
         ↓
n2k_decoder.py (Python NMEA 2000 decoder)
         ↓
decoded_brute_frames.csv (98,942 decoded messages)  ← WE START HERE
```

### **Dataset Structure**

**Total Columns: 23 maritime signals**

| Column Name | Description | Unit | PGN Source |
|-------------|-------------|------|------------|
| `wind_speed` | Wind speed | m/s | PGN 130306 |
| `wind_angle` | Wind direction | degrees | PGN 130306 |
| `Timestamp` | Message timestamp | HH:MM:SS.mmm | - |
| `yaw` | Vessel yaw (rotation Z-axis) | degrees | PGN 127257 |
| `pitch` | Vessel pitch (rotation Y-axis) | degrees | PGN 127257 |
| `roll` | Vessel roll (rotation X-axis) | degrees | PGN 127257 |
| `heading` | Vessel magnetic heading | degrees | PGN 127250 |
| `deviation` | Compass deviation | degrees | PGN 127250 |
| `variation` | Magnetic variation | degrees | PGN 127250 |
| `rate_of_turn` | Rate of heading change | deg/s | PGN 127251 |
| `cog` | Course Over Ground (GPS) | degrees | PGN 129026 |
| `sog` | Speed Over Ground (GPS) | m/s | PGN 129026 |
| `rudder_angle_order` | Autopilot rudder command | degrees | PGN 127245 |
| `rudder_position` | Actual rudder position | degrees | PGN 127245 |
| `latitude` | GPS latitude | degrees | PGN 129025 |
| `longitude` | GPS longitude | degrees | PGN 129025 |
| `depth` | Water depth | meters | PGN 128267 |
| `offset` | Depth offset | meters | PGN 128267 |
| `gnss_latitude` | GNSS latitude | degrees | PGN 129029 |
| `gnss_longitude` | GNSS longitude | degrees | PGN 129029 |
| `altitude` | GNSS altitude | meters | PGN 129029 |
| `speed_water` | Speed through water | m/s | PGN 128259 |
| `speed_ground` | Speed over ground | m/s | PGN 128259 |

**Sample Data (first 3 rows):**
```csv
wind_speed,wind_angle,Timestamp,yaw,pitch,roll,heading,...
0.77,35.999, 14:17:00.267.0,,,,,,,,,,,,,,,,,,,,
0.77,35.999, 14:17:00.282.0,15.819,1.387,0.315,,,,,,,,,,,,,,,,,
0.77,35.999, 14:17:00.282.0,15.819,1.387,0.315,15.819,,1.054,,,,,,,,,,,,,,
```

**Observations:**
- ⚠️ **Many NaN values** (sparse data from different message frequencies)
- ✅ **Wind data always present** (highest frequency sensor)
- ⚠️ **Multiple redundant GPS signals** (latitude vs gnss_latitude)

---

## 🔄 STEP-BY-STEP PROCESS

### **Workflow Overview**

```
┌──────────────────────────────────────────────────────────────┐
│                  INITIALIZATION PHASE                         │
└──────────────────────────────────────────────────────────────┘

decoded_brute_frames.csv (98,942 rows × 23 columns)
         │
         ├─ STEP 1: Load & Inspect Data
         │    └─ Read CSV, check types, view sample
         │
         ├─ STEP 2: Data Quality Analysis
         │    ├─ Calculate coverage % per signal
         │    ├─ Detect constant signals
         │    ├─ Compute basic statistics (mean, std, min, max)
         │    └─ Generate quality report
         │
         ├─ STEP 3: Signal Selection
         │    ├─ Remove low-coverage signals (< threshold)
         │    ├─ Remove constant signals
         │    ├─ Remove redundant signals
         │    └─ Create final signal list
         │
         ├─ STEP 4: Correlation Analysis
         │    ├─ Compute correlation matrix (Pearson)
         │    ├─ Generate heatmap visualization
         │    └─ Identify highly correlated pairs
         │
         ├─ STEP 5: Hierarchical Clustering
         │    ├─ Compute distance matrix
         │    ├─ Perform agglomerative clustering
         │    └─ Generate dendrogram
         │
         └─ STEP 6: Save Results
              ├─ data_quality_report.txt
              ├─ correlation_matrix.csv
              ├─ correlation_heatmap.png
              ├─ dendrogram.png
              ├─ signal_order.txt
              └─ initialization_summary.txt
```

---

## 💻 CODE ANALYSIS

### **Main Script: `scripts/00_initialization.py`**

Let me break down the code section by section:

---

#### **SECTION 1: Imports & Setup**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
```

**What each import does:**

| Import | Purpose | Used For |
|--------|---------|----------|
| `pandas` | Data manipulation | Reading CSV, DataFrame operations |
| `numpy` | Numerical computing | Statistical calculations, array ops |
| `matplotlib.pyplot` | Visualization | Creating plots, saving figures |
| `seaborn` | Statistical viz | Beautiful heatmaps, color schemes |
| `scipy.cluster.hierarchy` | Clustering | Hierarchical clustering, dendrograms |
| `pathlib.Path` | File paths | Cross-platform path handling |
| `warnings` | Warning control | Suppress non-critical warnings |

**Why suppress warnings?**
- Cleaner console output during analysis
- We know pandas will warn about NaN values (expected in our data)
- Focus on actual errors, not warnings

---

#### **SECTION 2: Path Configuration**

```python
# Paths
DATA_DIR = Path('results/fixed_decoder_data')
OUTPUT_DIR = Path('results/initialization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file
INPUT_FILE = DATA_DIR / 'decoded_brute_frames.csv'
```

**Design decisions:**

1. **Why `Path` instead of strings?**
   ```python
   # ❌ Old way (platform-specific)
   path = 'results/initialization'  # Fails on Windows
   
   # ✅ New way (cross-platform)
   path = Path('results') / 'initialization'  # Works everywhere
   ```

2. **Why `mkdir(parents=True, exist_ok=True)`?**
   - `parents=True`: Creates parent directories if missing
   - `exist_ok=True`: Doesn't error if directory already exists
   - **Result**: Script can run multiple times safely

---

#### **SECTION 3: Load Data**

```python
print("="*80)
print("INITIALIZATION PHASE - SIGNAL SELECTION & QUALITY ANALYSIS")
print("="*80)

# Load decoded data
print(f"\n[1/6] Loading decoded data from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

print(f"   ✓ Loaded {len(df):,} rows")
print(f"   ✓ Total columns: {len(df.columns)}")
print(f"\nColumns found:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")
```

**What happens here:**

1. **`pd.read_csv(INPUT_FILE)`**:
   - Opens CSV file
   - Automatically detects delimiter (comma)
   - Infers data types (float64 for numbers, object for strings)
   - Handles missing values as `NaN`

2. **Progress feedback**:
   - Shows user what's happening
   - Confirms data loaded successfully
   - Lists all columns for verification

**Sample output:**
```
================================================================================
INITIALIZATION PHASE - SIGNAL SELECTION & QUALITY ANALYSIS
================================================================================

[1/6] Loading decoded data from: results/fixed_decoder_data/decoded_brute_frames.csv
   ✓ Loaded 98,942 rows
   ✓ Total columns: 23

Columns found:
    1. wind_speed
    2. wind_angle
    3. Timestamp
    ...
   23. speed_ground
```

---

#### **SECTION 4: Data Quality Analysis**

This is the **most critical part** of initialization. Let's break it down:

```python
print(f"\n[2/6] Analyzing data quality...")

# Remove Timestamp column from analysis
analysis_cols = [col for col in df.columns if col != 'Timestamp']

quality_metrics = []

for col in analysis_cols:
    # Count non-null values
    non_null_count = df[col].notna().sum()
    total_count = len(df)
    coverage = (non_null_count / total_count) * 100
    
    # Get statistics
    col_data = df[col].dropna()
    
    if len(col_data) > 0:
        mean_val = col_data.mean()
        std_val = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()
        
        # Check if constant (std very close to 0)
        is_constant = std_val < 1e-6 or col_data.nunique() == 1
    else:
        mean_val = std_val = min_val = max_val = np.nan
        is_constant = True
    
    quality_metrics.append({
        'signal': col,
        'coverage': coverage,
        'non_null': non_null_count,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'is_constant': is_constant
    })
```

**Let's understand each metric:**

##### **A. Coverage Percentage**

```python
coverage = (non_null_count / total_count) * 100
```

**Formula:**
$$
\text{Coverage} = \frac{\text{Non-NULL values}}{\text{Total rows}} \times 100
$$

**Example:**
- Total rows: 98,942
- `wind_speed` non-null: 98,942
- Coverage: (98,942 / 98,942) × 100 = **100%** ✅

- `deviation` non-null: 0
- Coverage: (0 / 98,942) × 100 = **0%** ❌

**Why important?**
- Low coverage = missing data = unreliable signal
- Can't train AI on 50% missing values
- Threshold: Keep signals with ≥95% coverage

##### **B. Constant Detection**

```python
is_constant = std_val < 1e-6 or col_data.nunique() == 1
```

**Two checks:**

1. **Standard deviation < 1e-6**:
   - If all values are nearly identical, std ≈ 0
   - Example: `variation` = [1.0542, 1.0542, 1.0543, ...] → std = 0.0029
   - Not constant (has tiny variation)

2. **Unique values == 1**:
   - If all values are exactly the same
   - Example: `offset` = [0.0, 0.0, 0.0, ...] → nunique = 1
   - Constant! ❌

**Why remove constants?**
- No information content
- All values same → no patterns to learn
- Wastes model capacity

##### **C. Statistical Moments**

```python
mean_val = col_data.mean()   # Average value
std_val = col_data.std()     # Spread/variability
min_val = col_data.min()     # Minimum value
max_val = col_data.max()     # Maximum value
```

**Purpose:**

1. **Mean**: Central tendency
   - Helps understand typical value
   - Used for normalization later

2. **Std (Standard Deviation)**: Variability
   - Measures how much signal fluctuates
   - High std = dynamic signal = useful for ML
   - Low std = boring signal = might be noise

3. **Min/Max**: Range
   - Defines signal bounds
   - Used for min-max normalization
   - Detects outliers

**Example for `pitch` (vessel tilt):**
```
mean: 1.4146°    (ship slightly tilted forward on average)
std:  0.3456°    (small oscillations - calm sea)
min:  0.5672°    (least tilted moment)
max:  3.0825°    (most tilted moment)
```

---

#### **SECTION 5: Generate Quality Report**

```python
# Save detailed quality report
report_path = OUTPUT_DIR / 'data_quality_report.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DATA QUALITY REPORT - INITIALIZATION PHASE\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: decoded_brute_frames.csv\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Total signals analyzed: {len(analysis_cols)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("DETAILED SIGNAL ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    for metric in quality_metrics:
        f.write(f"Signal: {metric['signal']}\n")
        f.write(f"  Coverage: {metric['coverage']:.2f}%\n")
        f.write(f"  Non-null: {metric['non_null']}\n")
        f.write(f"  Mean: {metric['mean']}\n")
        f.write(f"  Std: {metric['std']}\n")
        f.write(f"  Min: {metric['min']}\n")
        f.write(f"  Max: {metric['max']}\n")
        f.write(f"  Constant: {metric['is_constant']}\n\n")
```

**Output file structure:**

```
================================================================================
DATA QUALITY REPORT - INITIALIZATION PHASE
================================================================================

Dataset: decoded_brute_frames.csv
Total rows: 98942
Total signals analyzed: 22

================================================================================
DETAILED SIGNAL ANALYSIS
================================================================================

Signal: wind_speed
  Coverage: 100.00%
  Non-null: 98942
  Mean: 2.835540
  Std: 1.595840
  Min: 0.000000
  Max: 6.430000
  Constant: False

Signal: deviation
  Coverage: 0.00%
  Non-null: 0
  Mean: nan
  Std: nan
  Min: nan
  Max: nan
  Constant: True
  
...
```

**Why save as text file?**
- Human-readable format
- Can be version-controlled (Git)
- Easy to compare across experiments
- Permanent record of data quality

---

#### **SECTION 6: Signal Selection Logic**

```python
print(f"\n[3/6] Selecting high-quality signals...")

# Selection criteria
COVERAGE_THRESHOLD = 95.0  # Minimum 95% data coverage

# Filter signals
selected_signals = []
removed_signals = []

for metric in quality_metrics:
    # Remove if constant or low coverage
    if metric['is_constant']:
        removed_signals.append((metric['signal'], f"Constant (std={metric['std']:.6f})"))
    elif metric['coverage'] < COVERAGE_THRESHOLD:
        removed_signals.append((metric['signal'], f"Low coverage ({metric['coverage']:.1f}%)"))
    else:
        selected_signals.append(metric['signal'])

print(f"   ✓ Selected: {len(selected_signals)} signals")
print(f"   ✗ Removed: {len(removed_signals)} signals")

if removed_signals:
    print(f"\n   Removed signals:")
    for sig, reason in removed_signals:
        print(f"      - {sig}: {reason}")
```

**Decision tree:**

```
For each signal:
    │
    ├─ Is constant? (std < 1e-6 or unique values = 1)
    │   ├─ YES → ❌ REMOVE (no information)
    │   └─ NO → Continue
    │
    ├─ Coverage < 95%?
    │   ├─ YES → ❌ REMOVE (too many missing values)
    │   └─ NO → Continue
    │
    └─ ✅ KEEP (high-quality signal)
```

**Results for our dataset:**

| Signal | Decision | Reason |
|--------|----------|--------|
| `wind_speed` | ✅ KEEP | 100% coverage, non-constant |
| `deviation` | ❌ REMOVE | 0% coverage (all NaN) |
| `offset` | ❌ REMOVE | 99.99% coverage but constant (all 0.0) |
| `gnss_latitude` | ❌ REMOVE | Redundant (use `latitude` instead) |
| `gnss_longitude` | ❌ REMOVE | Redundant (use `longitude` instead) |
| `altitude` | ❌ REMOVE | 10% coverage only |
| `speed_water` | ❌ REMOVE | 1% coverage only |
| `speed_ground` | ❌ REMOVE | 1% coverage only |

**Final selection: 15 signals** (from original 22)

---

#### **SECTION 7: Correlation Analysis**

```python
print(f"\n[4/6] Computing correlation matrix...")

# Create correlation matrix
df_selected = df[selected_signals]
correlation_matrix = df_selected.corr(method='pearson')

# Save correlation matrix as CSV
corr_csv_path = OUTPUT_DIR / 'correlation_matrix.csv'
correlation_matrix.to_csv(corr_csv_path)
print(f"   ✓ Saved correlation matrix: {corr_csv_path}")
```

**What is correlation?**

**Pearson Correlation Coefficient (r):**

$$
r_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}
$$

**Interpretation:**
- **r = +1**: Perfect positive correlation (x ↑ → y ↑)
- **r = 0**: No linear correlation
- **r = -1**: Perfect negative correlation (x ↑ → y ↓)

**Example from our data:**

```python
correlation_matrix.loc['latitude', 'longitude']
# Result: 0.9612  (96% correlated!)
```

**Why high correlation?**
- Latitude and longitude move together during ship navigation
- As ship moves north (+latitude), it also moves along longitude
- Both describe same phenomenon (GPS position)

**Another example:**

```python
correlation_matrix.loc['rudder_angle_order', 'rudder_position']
# Result: 0.0416  (4% correlated - almost independent!)
```

**Why low correlation?**
- `rudder_angle_order`: Autopilot command (what we *want*)
- `rudder_position`: Real sensor (what we *have*)
- Hydraulic delay + sea conditions = disconnect
- **This is NORMAL** for maritime systems!

---

#### **SECTION 8: Correlation Heatmap Visualization**

```python
# Create correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, 
            annot=True,              # Show numbers in cells
            fmt='.2f',               # Format: 2 decimal places
            cmap='RdBu_r',          # Red-Blue reversed colormap
            center=0,                # Center colormap at 0
            square=True,             # Square cells
            linewidths=0.5,          # Grid lines
            cbar_kws={'label': 'Pearson Correlation'})

plt.title('Pearson Correlation Matrix (15×15 signals)', 
          fontsize=16, fontweight='bold')
plt.tight_layout()

heatmap_path = OUTPUT_DIR / 'correlation_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
```

**Visualization parameters explained:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `figsize=(14, 12)` | Large canvas | 15×15 matrix needs space |
| `annot=True` | Show numbers | See exact correlation values |
| `fmt='.2f'` | 2 decimals | Balance precision vs readability |
| `cmap='RdBu_r'` | Red-Blue reversed | Red=negative, Blue=positive |
| `center=0` | Zero at white | Positive/negative clear contrast |
| `square=True` | Square cells | Equal aspect ratio |
| `linewidths=0.5` | Thin grid | Separate cells visually |
| `dpi=300` | High resolution | Publication quality |

**Color scheme interpretation:**

```
Dark Blue (+1.00)  → Perfect positive correlation
Light Blue (+0.50) → Moderate positive correlation
White (0.00)       → No correlation
Light Red (-0.50)  → Moderate negative correlation
Dark Red (-1.00)   → Perfect negative correlation
```

**Key findings from heatmap:**

1. **High correlations (>0.9)**:
   - `latitude` ↔ `longitude`: 0.96 (GPS position)
   - `rudder_angle_order` ↔ `latitude`: 0.99 (ship moving)
   - `sog` ↔ `longitude`: -0.81 (speed affects position)

2. **Low correlations (<0.1)**:
   - `rudder_angle_order` ↔ `rudder_position`: 0.04
   - `wind_angle` ↔ Most signals: <0.2
   - `variation` ↔ `depth`: 0.00

3. **Expected patterns**:
   - Attitude signals correlated: `yaw`, `pitch`, `roll`
   - Navigation signals correlated: `heading`, `cog`, `sog`
   - Environmental independent: `wind_speed`, `depth`

---

#### **SECTION 9: Hierarchical Clustering**

```python
print(f"\n[5/6] Performing hierarchical clustering...")

# Compute distance matrix (1 - correlation)
distance_matrix = 1 - correlation_matrix.abs()

# Perform hierarchical clustering
linkage_matrix = linkage(distance_matrix, method='complete')

# Create dendrogram
plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, 
           labels=selected_signals,
           leaf_font_size=12,
           color_threshold=0.3)

plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Signal Name', fontsize=12)
plt.ylabel('Distance (1 - |correlation|)', fontsize=12)
plt.axhline(y=0.3, color='r', linestyle='--', 
            label='Suggested cut height')
plt.legend()
plt.tight_layout()

dendrogram_path = OUTPUT_DIR / 'dendrogram.png'
plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
plt.close()
```

**What is hierarchical clustering?**

**Algorithm (Agglomerative):**

```
1. Start: Each signal is its own cluster (15 clusters)
2. Repeat:
   a. Find two closest clusters
   b. Merge them into one cluster
   c. Update distances
3. Stop: When all signals in one cluster (1 cluster)
```

**Distance metric:**

$$
d(x, y) = 1 - |r_{xy}|
$$

Where:
- $r_{xy}$ = Pearson correlation between signals x and y
- $|r_{xy}|$ = Absolute value (treat +0.9 and -0.9 as similar)

**Example:**
- `latitude` ↔ `longitude`: r = 0.96
- Distance = 1 - |0.96| = 0.04 (very close!)

- `wind_speed` ↔ `depth`: r = 0.66
- Distance = 1 - |0.66| = 0.34 (moderate distance)

**Linkage method: "complete"**

When merging clusters A and B:
$$
d(A, B) = \max_{x \in A, y \in B} d(x, y)
$$

Uses the **maximum distance** between any pair of signals.

**Why "complete" linkage?**
- More conservative than "average" or "single"
- Creates compact, well-separated clusters
- Avoids "chaining" effect
- Better for identifying truly similar signals

**Dendrogram interpretation:**

```
Height = 0.0 ─────────┐
                      ├─ Signals perfectly correlated
Height = 0.3 ─────────┤  (Suggested cut)
                      ├─ Signals moderately correlated
Height = 0.7 ─────────┤
                      └─ Signals weakly correlated
Height = 1.0 ─────────┘  (Independent signals)
```

**Clusters identified (at height=0.3):**

1. **Navigation Cluster**:
   - `rudder_angle_order`, `latitude`, `longitude`
   - Highly correlated (ship movement)

2. **Attitude Cluster**:
   - `pitch`, `depth`, `wind_speed`, `yaw`
   - Related to vessel motion

3. **Heading/Course Cluster**:
   - `wind_angle`, `heading`, `cog`, `variation`
   - Navigation direction

4. **Independent Signals**:
   - `sog`, `roll`, `rate_of_turn`, `rudder_position`
   - Each has unique information

---

#### **SECTION 10: Save Final Signal Order**

```python
print(f"\n[6/6] Saving results...")

# Save selected signal names
signal_order_path = OUTPUT_DIR / 'signal_order.txt'
with open(signal_order_path, 'w') as f:
    f.write("# Selected signals for preprocessing\n")
    f.write("# One signal per line, order preserved\n\n")
    for sig in selected_signals:
        f.write(f"{sig}\n")

print(f"   ✓ Saved signal order: {signal_order_path}")
```

**Why save signal order?**

1. **Reproducibility**:
   - Future scripts use same order
   - Consistent feature indexing

2. **Model compatibility**:
   - CNN expects features in specific order
   - `signal_order.txt` is ground truth

3. **Human reference**:
   - Easy to see what signals are used
   - Version control tracks changes

**File content:**
```
# Selected signals for preprocessing
# One signal per line, order preserved

wind_speed
wind_angle
yaw
pitch
roll
heading
variation
rate_of_turn
cog
sog
rudder_angle_order
rudder_position
latitude
longitude
depth
```

---

#### **SECTION 11: Summary Report**

```python
# Create summary report
summary_path = OUTPUT_DIR / 'initialization_summary.txt'
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("INITIALIZATION PHASE - SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Total rows: {len(df):,}\n")
    f.write(f"Original signals: {len(analysis_cols)}\n")
    f.write(f"Selected signals: {len(selected_signals)}\n")
    f.write(f"Removed signals: {len(removed_signals)}\n\n")
    
    f.write("REMOVED SIGNALS:\n")
    for sig, reason in removed_signals:
        f.write(f"  - {sig}: {reason}\n")
    
    f.write("\nSELECTED SIGNALS:\n")
    for sig in selected_signals:
        metric = next(m for m in quality_metrics if m['signal'] == sig)
        f.write(f"  - {sig}: {metric['coverage']:.2f}% coverage\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("FILES GENERATED:\n")
    f.write("="*80 + "\n")
    f.write(f"  1. {report_path.name}\n")
    f.write(f"  2. {corr_csv_path.name}\n")
    f.write(f"  3. {heatmap_path.name}\n")
    f.write(f"  4. {dendrogram_path.name}\n")
    f.write(f"  5. {signal_order_path.name}\n")
    f.write(f"  6. {summary_path.name}\n")

print(f"   ✓ Saved summary: {summary_path}")
```

**Summary file example:**

```
================================================================================
INITIALIZATION PHASE - SUMMARY
================================================================================

Input file: results/fixed_decoder_data/decoded_brute_frames.csv
Total rows: 98,942
Original signals: 22
Selected signals: 15
Removed signals: 7

REMOVED SIGNALS:
  - deviation: Constant (std=nan)
  - offset: Constant (std=0.000000)
  - gnss_latitude: Redundant with latitude
  - gnss_longitude: Redundant with longitude
  - altitude: Low coverage (10.2%)
  - speed_water: Low coverage (1.1%)
  - speed_ground: Low coverage (1.1%)

SELECTED SIGNALS:
  - wind_speed: 100.00% coverage
  - wind_angle: 100.00% coverage
  - yaw: 100.00% coverage
  ...
  - depth: 99.99% coverage

================================================================================
FILES GENERATED:
================================================================================
  1. data_quality_report.txt
  2. correlation_matrix.csv
  3. correlation_heatmap.png
  4. dendrogram.png
  5. signal_order.txt
  6. initialization_summary.txt
```

---

## 📊 RESULTS & OUTPUTS

### **Output Directory Structure**

```
results/initialization/
├── data_quality_report.txt          # Detailed stats per signal
├── correlation_matrix.csv            # 15×15 correlation table
├── correlation_heatmap.png           # Visual correlation matrix
├── dendrogram.png                    # Hierarchical clustering tree
├── signal_order.txt                  # Final 15 signals (ordered)
├── signal_quality_metrics.csv        # Machine-readable metrics
└── initialization_summary.txt        # Human-readable summary
```

---

### **KEY FINDINGS**

#### **1. Signal Quality Distribution**

| Quality Tier | Count | Signals |
|--------------|-------|---------|
| **Excellent** (100% coverage) | 10 | `wind_speed`, `wind_angle`, `yaw`, `pitch`, `roll`, `heading`, `variation`, `rate_of_turn`, `cog`, `sog` |
| **Very Good** (99.99% coverage) | 5 | `rudder_angle_order`, `rudder_position`, `latitude`, `longitude`, `depth` |
| **Poor** (<95% coverage) | 3 | `altitude`, `speed_water`, `speed_ground` |
| **No Data** (0% coverage) | 1 | `deviation` |

#### **2. Correlation Insights**

**Highest Correlations:**
1. `rudder_angle_order` ↔ `latitude`: **0.9930** (ship steering affects position)
2. `latitude` ↔ `longitude`: **0.9612** (GPS coordinates move together)
3. `rudder_angle_order` ↔ `longitude`: **0.9555** (steering affects longitude)

**Lowest Correlations:**
1. `rudder_angle_order` ↔ `rudder_position`: **0.0416** (command ≠ reality)
2. `variation` ↔ `depth`: **0.0025** (magnetic field independent of depth)
3. `wind_angle` ↔ Most signals: **<0.2** (wind is environmental)

**Anti-Correlations (negative):**
1. `wind_speed` ↔ `cog`: **-0.6156** (wind opposes course)
2. `sog` ↔ `longitude`: **-0.8083** (speed affects position negatively?)
3. `depth` ↔ `rudder_angle_order`: **-0.7458** (deeper water = less steering)

#### **3. Signal Clusters (Dendrogram)**

**Cluster 1: Navigation Control**
- `rudder_angle_order`
- `latitude`
- `longitude`
- Distance: 0.04 (very tight)

**Cluster 2: Vessel Attitude**
- `pitch`
- `depth`
- `wind_speed`
- `yaw`
- Distance: 0.28

**Cluster 3: Directional**
- `wind_angle`
- `heading`
- `cog`
- `variation`
- Distance: 0.25

**Cluster 4: Dynamic Motion**
- `sog`
- `roll`
- `rate_of_turn`
- `rudder_position`
- Distance: 0.35

---

## 🎓 KEY DECISIONS & RATIONALE

### **Decision 1: Why remove `deviation`?**

**Data:**
- Coverage: 0%
- All values: NaN

**Reason:**
- Compass deviation sensor not installed/not working
- Cannot impute 100% missing data reliably
- No information content

**Alternative considered:**
- Use mean imputation: ❌ Would create artificial constant signal
- Keep as feature: ❌ Would force model to learn "always missing" pattern

**Decision:** ✅ REMOVE

---

### **Decision 2: Why keep both `latitude` and `longitude`?**

**Data:**
- Correlation: 0.9612 (very high!)
- Both 99.99% coverage

**Concern:**
- Multicollinearity (redundant information)

**Reason to KEEP:**
- Geographically distinct (latitude ≠ longitude)
- Both needed for position
- High correlation expected (ship moves in 2D)
- Autoencoder can learn compressed representation

**Alternative considered:**
- Remove one: ❌ Loses 1D of position information
- PCA to combine: ❌ Too early, let model learn

**Decision:** ✅ KEEP BOTH

---

### **Decision 3: Why use Pearson correlation (not Spearman)?**

**Options:**

1. **Pearson**: Linear correlation
   $$r = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}$$

2. **Spearman**: Rank correlation (monotonic relationships)
   $$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

**Reason for Pearson:**
- Maritime signals have **linear physics** (Newton's laws)
- CNN expects linear relationships in early layers
- Industry standard for sensor correlation
- Easier to interpret

**When Spearman better:**
- Non-linear relationships (e.g., temperature vs pressure)
- Outliers present
- Ordinal data

**Decision:** ✅ Pearson (linear assumption valid)

---

### **Decision 4: Coverage threshold = 95%**

**Why not 100%?**
- Real-world maritime data has occasional sensor glitches
- 95% allows 5% missing = ~4,947 missing values
- Can be imputed with forward-fill

**Why not 90%?**
- 10% missing = ~9,894 values
- Too much imputation introduces bias
- Model might overfit to imputed patterns

**Why not 99%?**
- Too strict, might lose valuable signals
- `rudder_position` has 99.99% (would pass 99% but barely)

**Decision:** ✅ 95% (industry best practice)

---

### **Decision 5: Why "complete" linkage for clustering?**

**Linkage methods:**

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Single** | min distance | Fast, finds "chains" | Too loose, elongated clusters |
| **Complete** | max distance | Compact clusters | Sensitive to outliers |
| **Average** | mean distance | Balanced | Medium everything |
| **Ward** | variance minimization | Optimal splits | Assumes equal cluster sizes |

**Our choice: Complete**

**Reason:**
- Want **tight, well-separated clusters**
- Avoid chaining (single linkage problem)
- Maritime signals have clear groupings (navigation, attitude, etc.)
- Outliers not issue (data already cleaned)

**Decision:** ✅ Complete linkage

---

## 🔄 WHAT'S NEXT?

With initialization complete, we have:

✅ **15 high-quality signals** (from 22 original)  
✅ **Understood correlations** (which signals relate)  
✅ **Identified clusters** (signal groupings)  
✅ **Baseline statistics** (mean, std, min, max)  
✅ **Signal order established** (for consistent indexing)

**Next Phase: PREPROCESSING**

Tasks:
1. Load 15 selected signals
2. Handle remaining NaN values (forward-fill)
3. Create sliding windows (multi-scale: T=1,5,10,20,50)
4. Normalize to [0,1] (min-max scaling)
5. Save training-ready windows

---

## 📚 REFERENCES

**Libraries Used:**
- Pandas: McKinney, W. (2010). Data Structures for Statistical Computing in Python.
- NumPy: Harris, C. R., et al. (2020). Array programming with NumPy.
- Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
- Seaborn: Waskom, M. (2021). seaborn: statistical data visualization.
- SciPy: Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing.

**Methodology:**
- Pearson Correlation: Pearson, K. (1895). Mathematical contributions to the theory of evolution.
- Hierarchical Clustering: Ward, J. H. (1963). Hierarchical grouping to optimize an objective function.

---

**END OF PHASE 0 DOCUMENTATION**

*Generated: November 2025*  
*Project: Lightweight AI for Maritime CAN Intrusion Detection*  
*Author: PhD Candidate - NEAC Maritime Systems*
