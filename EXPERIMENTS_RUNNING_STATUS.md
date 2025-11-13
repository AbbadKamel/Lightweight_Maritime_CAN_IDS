# 🚀 ALL EXPERIMENTS RUNNING - Status Report

**Date:** November 13, 2025  
**Time Started:** 11:21 CET (Old approach) | 11:42 CET (New approach)  
**Location:** N315L-G17G01.ressource.unicaen.fr  

---

## 📊 Overview

We are running **7 parallel experiments** comparing two different detection methodologies:

### 🔴 OLD APPROACH: Simple MSE Averaging + Voting (4 experiments)
- **Started:** 11:21 CET (22 minutes ago)
- **Status:** 🟡 Loading models and data
- **CPU Usage:** 509-551% per process (5-6 cores each)
- **RAM Usage:** 16-16.4 GB per process
- **Total RAM:** ~65 GB

### 🟢 NEW APPROACH: CANShield 3-Tier Hierarchical Detection (3 experiments)
- **Started:** 11:42 CET (1 minute ago)
- **Status:** 🟡 Loading models and data
- **CPU Usage:** 319-354% per process (3-4 cores each)
- **RAM Usage:** 17-17.3 GB per process
- **Total RAM:** ~51 GB

**Total System Load:** ~116 GB RAM, ~3000% CPU (30 cores) ✅ Well within capacity!

---

## 📋 Experiment Details

### OLD APPROACH (Simple MSE Averaging)

These experiments use the **FLAWED methodology** we initially implemented:
- Calculate MSE for each of 5 autoencoders
- Average MSE across autoencoders
- Apply voting threshold (N out of 5 must agree)
- Use single global percentile threshold

| Experiment | Voting | Threshold | PID | CPU | Expected Result |
|------------|--------|-----------|-----|-----|-----------------|
| exp_vote1_p95 | 1/5 | p95 | 209234 | 534% | Very high recall, very high FP |
| **exp_vote2_p95** | **2/5** | **p95** | 209201 | 545% | **98% recall, 97% FP (best)** |
| exp_vote2_p99 | 2/5 | p99 | 209227 | 509% | Lower recall, maybe better precision |
| exp_vote3_p95 | 3/5 | p95 | 209219 | 551% | 21% recall, 35% precision (baseline) |

### NEW APPROACH (CANShield Hierarchical)

These experiments use the **CORRECT CANShield methodology**:
- **Tier 1:** Per-signal absolute error thresholds (15 independent thresholds)
- **Tier 2:** Temporal anomaly counting (% of timesteps anomalous per signal)
- **Tier 3:** Multi-signal voting (% of signals detecting anomaly)

| Experiment | Loss Factor | Time Factor | Signal Factor | PID | CPU | Expected Result |
|------------|-------------|-------------|---------------|-----|-----|-----------------|
| **canshield_95_95_95** | **p95** | **p95** | **p95** | 1175617 | 354% | **Balanced (most likely best)** |
| canshield_99_99_99 | p99 | p99 | p99 | 1177281 | 319% | Very strict, high precision |
| canshield_95_99_95 | p95 | p99 | p95 | 1178780 | 332% | Strict temporal (good for flooding) |

---

## ⏱️ Estimated Timeline

### Old Approach (Started 11:21):
- **Loading phase:** 20 minutes (until ~11:41)
- **Processing phase:** 30-40 minutes
- **Expected completion:** ~12:00-12:10 CET

### New Approach (Started 11:42):
- **Loading phase:** 20 minutes (until ~12:02)
- **Processing phase:** 30-40 minutes (might be faster due to different calculation)
- **Expected completion:** ~12:20-12:30 CET

**All experiments should complete by:** ~12:30 CET (45 minutes from now)

---

## 🎯 What We're Testing

### Key Question:
**Does the CANShield 3-tier hierarchical detection solve the precision problem?**

### Old Approach Problems:
- ❌ Uses MSE instead of absolute error
- ❌ Global threshold instead of per-signal thresholds
- ❌ No temporal anomaly counting
- ❌ Simple voting can't distinguish attack patterns from normal variations
- ❌ Result: Either low recall OR high false positives

### New Approach Advantages:
- ✅ Uses absolute error (more sensitive to anomalies)
- ✅ Per-signal thresholds (detects which signals are anomalous)
- ✅ Temporal counting (flooding = many repeated anomalies)
- ✅ Hierarchical detection (3 levels of validation)
- ✅ Expected: High recall AND high precision

---

## 📈 Expected Results Comparison

### Old Approach (Best case: vote2_p95):
```
Recall:    ~98%
Precision: ~30% (97% false positives!)
F1-Score:  ~46%
```

### New Approach (Expected: canshield_95_95_95):
```
Recall:    >90% (based on CANShield paper)
Precision: >90% (based on CANShield paper)
F1-Score:  >90%
```

---

## 🔍 Monitoring Commands

### Check all experiments status:
```bash
bash check_all_experiments.sh
```

### SSH to supercomputer:
```bash
ssh abbad241@N315L-G17G01.ressource.unicaen.fr
```

### Attach to experiment (inside SSH):
```bash
screen -r exp_vote2_p95         # Old approach - best config
screen -r canshield_95_95_95    # New approach - balanced config
```

### Check logs (inside SSH):
```bash
cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS/logs
tail -f exp_vote2_p95.log
tail -f canshield_95_95_95.log
```

### Check results (when complete):
```bash
cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
ls -lh results_*.json
python3 compare_experiments.py  # Compare all 7 results!
```

---

## 📊 Dataset Information

- **Total samples:** 4,983,120 (4.9M)
- **Normal traffic:** 3,452,268 (69.3%)
- **Attack traffic:** 1,530,852 (30.7%)

### Attack Breakdown:
- **DoS:** 480,852 samples (9.65%)
- **Fuzzing:** 490,000 samples (9.83%)
- **Spoofing:** 490,000 samples (9.83%)
- **Suppress:** 70,000 samples (1.40%)

### Attack Characteristics:
- **Strength:** 90-100% corruption
- **Pattern:** Square wave, continuous, flooding (50-100 repeats)
- **All 15 signals** can be attacked

---

## 🎬 Next Steps (When Experiments Complete)

1. **Run comparison script:**
   ```bash
   ssh abbad241@N315L-G17G01.ressource.unicaen.fr
   cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
   python3 compare_experiments.py
   ```

2. **Analyze results:**
   - Compare precision-recall trade-offs
   - Identify best configuration(s)
   - Understand why hierarchical detection works better (or not)

3. **Decision for paper:**
   - If CANShield ≥90% precision AND recall: Use it! ✅
   - If both approaches similar: Document trade-offs, choose based on interpretability
   - If old approach better: Investigate why (unlikely based on theory)

4. **Download results:**
   ```bash
   scp abbad241@N315L-G17G01.ressource.unicaen.fr:/data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS/results_*.json .
   ```

5. **Create visualizations:**
   - Confusion matrices
   - ROC curves
   - Per-attack detection rates
   - Computational cost comparison

---

## ✅ Success Criteria

**Minimum acceptable results:**
- Recall ≥ 85%
- Precision ≥ 85%
- F1-Score ≥ 85%

**Target results (for strong paper):**
- Recall ≥ 95%
- Precision ≥ 95%
- F1-Score ≥ 95%

**CANShield paper reported:**
- Recall: ~99%
- Precision: ~99%
- F1-Score: ~99%

---

## 🚨 Potential Issues to Watch

1. **Memory overflow:** Each process uses ~17GB, total ~116GB < 256GB available ✅
2. **CPU contention:** Using ~30 cores out of 40 available ✅
3. **Disk I/O:** 148MB dataset loaded 7 times = ~1GB total, negligible ✅
4. **Process crashes:** Monitor logs for errors ⚠️
5. **Network timeout:** SSH sessions might disconnect (but screen keeps running) ✅

---

## 📝 Notes

- The old approach will likely confirm the precision problem we already know about
- The new CANShield approach should dramatically improve precision while maintaining recall
- If CANShield works, we can cite the methodology and show our implementation on maritime CAN data
- All experiments run independently - no interference between them
- Results will be saved as JSON files for easy comparison

**Status:** 🟢 ALL SYSTEMS GO! Waiting for results...

---

**Last Updated:** November 13, 2025, 11:43 CET  
**Next Check:** 12:00 CET (old approach should be processing by then)  
**Final Results Expected:** 12:30 CET
