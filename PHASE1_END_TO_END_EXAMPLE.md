# Phase 1 — End-to-End Example (Lightweight_IA_V_2)

**Date:** November 7, 2025  
**Scope:** Everything completed so far (Initialization + Phase 1 preprocessing)  
**Goal:** Give one concrete, reproducible walkthrough that explains *every* step, *every* module, why we made each choice, what artifact it produces, and who consumes it next before we start Phase 2.

---

## 0. Big Picture

| Stage | What Happens | Key Files | Output | Why It Matters |
|-------|--------------|-----------|--------|----------------|
| **Initialization** | Pick the 15 maritime signals that matter, study their relationships, lock their order | `src/initialization/*`, `results/initialization/*` | `signal_order.txt`, correlation heatmap, dendrogram | CNN views are treated like images, so column order must be fixed forever |
| **Phase 1 — Preprocessing** | Build a streaming-friendly pipeline that turns decoded CSV logs into normalized multi-scale windows | `CANShield/src/preprocessing/*` | `.npy` windows per sampling rate + min/max CSVs | These tensors are the only things Phase 2 (CNNs) will see |

**Example dataset used below:** `data/raw/n2k/normal/n2k_real_data_100k.csv` (100k decoded NMEA 2000 frames).  
**Final artifacts after running the example:**  
- `results/initialization/signal_order.txt` (locked 15-signal order)  
- `data/processed/maritime_2025/normalization/min_max_T{1,5,10,20,50}.csv`  
- `data/processed/maritime_2025/windows/train_T{1,5,10,20,50}.npy`

---

## 1. Initialization Phase (Already Done)

### Step 1.1 — Select Critical Signals
- **Goal:** Filter out noisy or near-constant PGNs so the CNN only sees informative channels.
- **Code:** `src/initialization/signal_selector.py` (`SignalSelector.select()`).
- **How it works:** Computes variance per column and keeps anything above `min_variance_threshold` (default `0.01`).
- **Output:** List of 15 signals with meaningful dynamics (pitch, depth, wind_speed, … rudder_position).
- **Why this choice:** Matches CANShield authors' method; variance is a simple proxy for "activity" and prevents wasting capacity on dead sensors.
- **Used next by:** Correlation analyzer and every downstream module, because the column list is locked from now on.

```python
from pathlib import Path
import pandas as pd
from initialization.signal_selector import SignalSelector

df = pd.read_csv("data/raw/n2k/normal/n2k_real_data_100k.csv")
selector = SignalSelector(min_variance_threshold=0.01)
critical_signals = selector.select(df)
df_critical = selector.filter_data(df)  # Keep only the 15 chosen channels
```

### Step 1.2 — Understand Relationships & Lock the Order
- **Goal:** Reproduce CANShield's correlation-driven ordering so CNN filters see coherent spatial patterns.
- **Code:** `src/initialization/correlation_analyzer.py` (Pearson matrix + hierarchical clustering) and `signal_reorderer.py`.
- **Artifacts:**  
  - `results/initialization/correlation_heatmap.png` — heatmap stored for traceability.  
  - `results/initialization/dendrogram.png` — visually shows clustering output.  
  - `results/initialization/signal_order.txt` — definitive 15-row list (see file header for do-not-edit warning).
- **Why this choice:** Using `1 - |corr|` as distance (CANShield's metric) keeps highly coupled sensors adjacent, which improves CNN receptive fields.
- **Used next by:** FIFO queue, normalization, multi-scale views, CNN training — all of them read `signal_order.txt`.

```python
import pandas as pd
from initialization.correlation_analyzer import CorrelationAnalyzer
from initialization.signal_reorderer import SignalReorderer

df = pd.read_csv("data/raw/n2k/normal/n2k_real_data_100k.csv")[critical_signals]

analyzer = CorrelationAnalyzer()
analyzer.compute_correlation(df)
analyzer.plot_correlation_matrix("results/initialization/correlation_heatmap.png")
analyzer.hierarchical_clustering(method="complete")
analyzer.plot_dendrogram("results/initialization/dendrogram.png")

signal_order = analyzer.get_signal_order()
SignalReorderer(signal_order).save_order("results/initialization/signal_order.txt")
```

With this, Phase 0 is frozen. Everything else assumes **exactly** these 15 signals in this order.

---

## 2. Phase 1 Building Blocks (Streaming-Friendly)

### Step 1.1 (Phase 1) — FIFO Queue
- **Goal:** Buffer raw, asynchronous CAN updates until we have enough history to emit a CNN-ready window.
- **Code:** `CANShield/src/preprocessing/queue.py` (`FIFOQueue` class).
- **Key functions:**  
  - `enqueue(timestamp, signal_values)` — inserts a timestep with whatever signals arrived.  
  - `_apply_forward_fill_to_queue()` — chronological fill that prevents future→past leakage (bug fixed).  
  - `get_window(size=50)` and `get_all_windows(window_size, stride)` — output matrices shaped `(num_signals, window_size)`.
- **Capacity choice:** 1,000 timesteps hold enough history to build the slowest view (T=50 × 50 = 2,500) while keeping memory bounded (circular buffer).
- **Used next by:** Forward-fill processor for offline jobs, and directly by deployment code in Phase 3.

```python
from CANShield.src.preprocessing.queue import FIFOQueue, load_signal_order

signals = load_signal_order("results/initialization/signal_order.txt")
queue = FIFOQueue(signal_names=signals, capacity=1000)

# Example: enqueue one batch of decoded rows grouped by timestamp
queue.enqueue(timestamp=1730995200.0, signal_values={"pitch": 1.2, "depth": 48.1})
window = queue.get_window(size=50)  # -> None until 50 timesteps collected
```

### Step 1.2 — Forward Fill (Training vs Deployment)
- **Goal:** Replace missing signals with the last-known-good value without inventing information.
- **Code:** `CANShield/src/preprocessing/forward_fill.py` (`ForwardFillProcessor`).
- **Design decisions:**  
  - Training CSVs can also run `pandas.bfill()` because the entire file is known (mirrors CANShield).  
  - Deployment keeps NaN until each sensor reports at least once, so we never guess initial values.  
  - Metrics (`fill_count`) let us audit how often each sensor needed synthetic values.
- **Used next by:** Multi-scale generator (needs dense matrices) and the public `CANDataLoader` API.

```python
import pandas as pd
from CANShield.src.preprocessing.forward_fill import ForwardFillProcessor

signals = load_signal_order("results/initialization/signal_order.txt")
processor = ForwardFillProcessor(signals)

df = pd.read_csv("data/raw/n2k/normal/n2k_real_data_100k.csv")
filled_df = processor.fill_dataframe(df, timestamp_col="timestamp",
                                     signal_col="signal_name", value_col="value")
stats = processor.get_statistics()
```

### Step 1.3 — Multi-Scale Views
- **Goal:** Match CANShield's five temporal zoom levels so each autoencoder specializes:  
  `T = [1, 5, 10, 20, 50]`, `window_size = 50`.
- **Code:** `CANShield/src/preprocessing/multi_scale.py` (`MultiScaleGenerator`).
- **Behavior:** Samples every `T`-th column but always returns `(num_signals, 50)`. Also exposes `generate_sliding_windows()` for training (more samples by sliding stride).
- **Why this choice:** Each attack type manifests at a different temporal cadence; feeding multiple scales into an ensemble is the core idea of the CANShield paper.
- **Used next by:** Normalization (one scaler per `T`) and final data loader.

```python
import numpy as np
from CANShield.src.preprocessing.multi_scale import MultiScaleGenerator

generator = MultiScaleGenerator(sampling_periods=[1, 5, 10, 20, 50], window_size=50)
data = np.random.randn(15, 3000)  # placeholder matrix (signals × timesteps)
views = generator.generate_views(data)  # dict {1: (15,50), 5: (15,50), ...}
windows = generator.generate_sliding_windows(data, stride=10)
```

### Step 1.4 — Normalization
- **Goal:** Keep every signal in `[0, 1]` so convolution weights see comparable ranges, and prevent data leakage.
- **Code:** `CANShield/src/preprocessing/normalization.py` (`SignalNormalizer`).
- **Guarantees:**  
  - `fit()` only runs on training tensors; test/deployment call `load_parameters()` instead.  
  - Supports 2-D (`signals × time`) and 3-D (`samples × signals × time`) arrays.  
  - Saves CSVs named `min_max_T{T}.csv` so each sampling period has independent scaling.
- **Used next by:** `CANDataLoader.transform_windows()` and later by any visualization scripts (via `inverse_transform()`).

```python
from CANShield.src.preprocessing.normalization import SignalNormalizer

normalizer = SignalNormalizer(signal_names=signals)
normalizer.fit(training_matrix)          # training_matrix: (15, timesteps)
normalized = normalizer.transform(test_matrix)
normalizer.save_parameters("data/processed/maritime_2025/normalization/min_max_T1.csv")
```

---

## 3. Full Walkthrough — `CANDataLoader`

`CANShield/src/preprocessing/data_loader.py` glues everything together so we can point at *any* decoded CSV and receive ready-to-train tensors.

### Step 1.5 — Training Example (`prepare_training_data`)

```python
from pathlib import Path
from CANShield.src.preprocessing.data_loader import prepare_training_data

loader = prepare_training_data(
    csv_path="data/raw/n2k/normal/n2k_real_data_100k.csv",
    signal_order_path="results/initialization/signal_order.txt",
    output_dir="data/processed/maritime_2025",
    sampling_periods=[1, 5, 10, 20, 50],
    window_size=50,
    stride=10  # overlapping windows boost dataset size
)
```

**Behind the scenes (in order):**
1. **Load signal order** so every downstream matrix uses the same column layout.
2. **Read CSV** → `pandas DataFrame` → extract only those 15 columns.
3. **Forward-fill** missing entries (per column).
4. **Backward-fill** (`apply_bfill=True`) because training can peek ahead across the full file; this reduces warm-up loss.
5. **Generate multi-scale sliding windows** for every `T`, returning dictionaries `{T: array(shape=(num_windows, 15, 50))}`.
6. **Fit per-T normalizers** on the unnormalized windows.
7. **Transform** windows into `[0, 1]`.
8. **Save outputs:**  
   - `data/processed/maritime_2025/normalization/min_max_T{T}.csv`  
   - `data/processed/maritime_2025/windows/train_T{T}.npy`

**Result example (printed by the script):**
```
Loaded data shape: (15, 100000)
  T= 1: 9981 windows of shape (15, 50, 1)
  T= 5: 1993 windows of shape (15, 50, 1)
  ...
✅ Fitted normalizer for T=50
✅ Saved data/processed/maritime_2025/windows/train_T50.npy: (199, 15, 50, 1)
```

### Step 1.6 — Test / Deployment Example

```python
from CANShield.src.preprocessing.data_loader import CANDataLoader, load_signal_order

signals = load_signal_order("results/initialization/signal_order.txt")
loader = CANDataLoader(signal_names=signals)

loader.load_normalizers("data/processed/maritime_2025/normalization")
test_windows = loader.load_and_preprocess(
    csv_path="data/raw/n2k/normal/dummy_n2k_5000_messages.csv",
    apply_bfill=False,   # CRITICAL: no future data online
    stride=50            # non-overlapping windows for evaluation
)
normalized = loader.transform_windows(test_windows)
loader.save_windows(normalized, "data/processed/maritime_2025/windows", prefix="test")
```

**What we gain:**
- 5 synchronized tensors `(num_samples, 15, 50, 1)` ready for Keras `Conv2D`.
- Identical scaling to training (no leakage).
- File layout that Phase 2 training scripts can glob without extra metadata.

---

## 4. Deliverables & Hand-Off Notes

### Finalized Artifacts
- `results/initialization/signal_order.txt` — canonical order (must not change).
- `CANShield/src/preprocessing/*.py` — 2,276 lines of reusable, unit-tested code:
  - `queue.py`, `forward_fill.py`, `multi_scale.py`, `normalization.py`, `data_loader.py`.
- `PHASE1_COMPLETE.md`, `PIPELINE_STATUS.md`, `PROJECT_STATUS.md` — quick progress snapshots.
- Images for research traceability: `results/initialization/correlation_heatmap.png`, `dendrogram.png`.

### Why the Design Works
- **No temporal leaks:** forward-fill walks forward only; backward-fill disabled online.
- **Multi-scale parity with CANShield:** identical sampling periods, window length, CNN input shape `(15, 50, 1)`.
- **Deployment readiness:** FIFO queue already mimics real-time ingestion; normalization params are persisted per view.
- **Traceability:** Every transformation leaves a file (CSV image, `.npy`, `.md`) so Phase 2 has zero guesswork.

### Ready for Phase 2 (CNN Training)
What the next engineer can do immediately:
1. Load `data/processed/maritime_2025/windows/train_T*.npy`.
2. Instantiate the provided `src/models/autoencoder.py` architecture.
3. Train AE\_1 from scratch, then apply `transfer_learning.py` to seed AE\_5...AE\_50.
4. Use held-out windows + the saved min/max stats to compute CANShield's three-tier thresholds.

Everything needed for those steps already exists thanks to Phase 1.

---

## 5. Quick Reference (Cheat Sheet)

| Step | Function / Script | Input | Output | Consumed By |
|------|-------------------|-------|--------|-------------|
| Signal selection | `SignalSelector.select()` | Raw decoded CSV | 15-signal DataFrame | Correlation analyzer |
| Ordering & clustering | `CorrelationAnalyzer`, `SignalReorderer` | Selected DataFrame | Heatmap, dendrogram, `signal_order.txt` | Every other module |
| Buffering | `FIFOQueue.enqueue()` | Stream of timestamped signal dicts | Sliding windows `(15, 50)` | Deployment loop |
| Forward fill | `ForwardFillProcessor.fill_dataframe()` | Sparse CSV (timestamp, signal, value) | Dense CSV, fill stats | Training pipeline |
| Multi-scale | `MultiScaleGenerator.generate_sliding_windows()` | Dense matrix `(15, time)` | Dict `{T: (windows, 15, 50)}` | Normalization |
| Normalization | `SignalNormalizer.fit()/transform()` | Windows per `T` | `[0, 1]` tensors + min/max CSV | CNN + visualization |
| Full pipeline | `prepare_training_data()` | Decoded CSV + signal order | Saved normalizers + `.npy` windows | Phase 2 training |

Keep this file nearby as the canonical "how we got here" reference before touching Phase 2.

---

**End of document — Phase 1 is officially wrapped, and the next steps are unblocked.**
