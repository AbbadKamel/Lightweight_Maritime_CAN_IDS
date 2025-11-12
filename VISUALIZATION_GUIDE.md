# 📊 Visualization Guide

## ✅ What Just Happened?

You now have **10 visualization files** showing your preprocessing pipeline!

---

## 📁 Generated Visualizations

All plots saved to: `results/visualizations/`

### **1. Raw Signal Time Series** (`01_raw_signals.png`)
- **What it shows:** All 15 CAN signals plotted over time
- **Purpose:** See raw data patterns, trends, and anomalies
- **Details:** 
  - Each signal in separate subplot
  - Red dashed line = mean value
  - Statistics box shows μ (mean) and σ (standard deviation)

### **2. Normalization Effect** (`02_normalization_effect.png`)
- **What it shows:** Before/after normalization comparison
- **Purpose:** Verify normalization works correctly
- **Details:**
  - Left column: Original values with real ranges
  - Right column: Normalized [0, 1] values
  - Shows first 500 timesteps
  - First 6 signals displayed

### **3. Distribution Analysis** (`03_distribution_*.png`)
- **What it shows:** Histogram and box plot of signal distributions
- **Purpose:** Understand signal value distributions
- **Files created:**
  - `03_distribution_pitch.png`
  - `03_distribution_depth.png`
  - `03_distribution_wind_speed.png`
- **Details:**
  - Left: Density histogram comparing raw vs normalized
  - Right: Box plot showing quartiles and outliers

### **4. Multi-Scale Comparison** (`04_multiscale_*.png`)
- **What it shows:** Same signal at different sampling rates
- **Purpose:** Visualize multi-scale temporal patterns
- **Files created:**
  - `04_multiscale_pitch.png`
  - `04_multiscale_depth.png`
  - `04_multiscale_wind_speed.png`
- **Details:**
  - 5 rows, one per sampling period (T=1, 5, 10, 20, 50)
  - T=1: Every timestep (fast patterns)
  - T=50: Every 50th timestep (slow trends)
  - Markers show actual sampled points

### **5. Sample Windows** (`05_sample_windows.png`)
- **What it shows:** 4 example windows as heatmaps
- **Purpose:** See what CNN will receive as input
- **Details:**
  - 4 windows from T=1 sampling period
  - Heatmap: Green=high value, Red=low value
  - Rows = signals, Columns = timesteps
  - All normalized to [0, 1]

### **6. Single Window Heatmap** (`06_single_window_heatmap.png`)
- **What it shows:** Detailed view of one window
- **Purpose:** Inspect individual CNN input
- **Details:**
  - Shape: (15 signals × 50 timesteps)
  - Color intensity = normalized value
  - This is exactly what autoencoder sees

---

## 🔍 How to View Them

### **Option 1: File Manager**
```bash
cd results/visualizations/
xdg-open 01_raw_signals.png  # Opens with default image viewer
```

### **Option 2: VS Code**
1. Open VS Code Explorer
2. Navigate to `results/visualizations/`
3. Click any `.png` file
4. VS Code will display it

### **Option 3: Command Line (if you have display)**
```bash
cd results/visualizations/
eog *.png  # Eye of GNOME image viewer
# or
display *.png  # ImageMagick
```

---

## 🚀 Using With Your Real Data

**To visualize YOUR actual CAN data:**

1. **Make sure you have:**
   - Decoded CAN CSV file
   - Signal order file from initialization

2. **Edit `example_visualize.py`:**
   ```python
   # Line 42-43: Change these paths
   csv_path = 'path/to/your/decoded_CAN_data.csv'
   signal_order_path = 'results/initialization/signal_order.txt'
   ```

3. **Run it:**
   ```bash
   python3 example_visualize.py
   ```

4. **Check results:**
   ```bash
   ls -lh results/visualizations/
   ```

---

## 📊 What Each Visualization Tells You

### **Quality Checks:**

✅ **Raw Signals** - Should show:
- Realistic ranges for maritime signals
- No constant values (flat lines)
- Some variation over time

✅ **Normalization** - Should show:
- All normalized values between 0 and 1
- Original patterns preserved (just scaled)
- No distortion of signal shapes

✅ **Distributions** - Should show:
- Normalized distributions centered around 0.5
- Raw distributions match expected sensor ranges
- No extreme outliers (unless real)

✅ **Multi-Scale** - Should show:
- Different temporal patterns at each scale
- T=1 captures fast changes
- T=50 captures slow trends
- Same overall pattern visible in all

✅ **Windows** - Should show:
- Heatmaps with varied colors (not all one color)
- Patterns across signals (correlated signals similar)
- Shape (15, 50, 1) ready for CNN

---

## 🎨 Customization

### **Change number of signals shown:**
```python
# In example_visualize.py, line 194
for idx in range(min(3, len(signal_names))):  # Change 3 to any number
```

### **Change sampling periods:**
```python
# Line 220
generator = MultiScaleGenerator(
    sampling_periods=[1, 5, 10, 20, 50],  # Modify this list
    window_size=50
)
```

### **Change output directory:**
```python
# Line 44
output_dir = 'my_custom_plots/'
```

---

## 🐛 Troubleshooting

### **"FileNotFoundError: CSV not found"**
- Script creates synthetic demo data automatically
- To use real data, update `csv_path` variable

### **"ModuleNotFoundError: matplotlib"**
```bash
pip install matplotlib seaborn
```

### **"No display found"**
- Script uses `Agg` backend (no display needed)
- All plots save to files automatically
- View files afterward

### **Plots look weird**
- Check data quality (NaN values, outliers)
- Verify signal_order.txt matches CSV columns
- Ensure sufficient data (>2500 timesteps for T=50)

---

## 📈 Next Steps

Now that you can visualize your data:

1. **✅ Verify data quality** - Check raw signals look correct
2. **✅ Confirm normalization** - All values [0, 1]
3. **✅ Inspect windows** - CNN input looks reasonable
4. **✅ Ready for Phase 2** - Train autoencoders!

---

## 🎯 Quick Reference Commands

```bash
# Generate visualizations
python3 example_visualize.py

# View all plots
cd results/visualizations/
ls -lh

# Open specific plot
xdg-open 01_raw_signals.png

# Copy plots to another location
cp results/visualizations/*.png ~/Desktop/
```

---

**Status:** Visualization tools ready! 📊  
**Output:** 10 PNG files in `results/visualizations/`  
**Next:** Inspect plots, verify quality, proceed to CNN training
