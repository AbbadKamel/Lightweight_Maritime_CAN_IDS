# Lightweight_IA_V_2

**Lightweight Intrusion Detection for Automotive CAN Networks**  
Based on CANShield (IEEE IoT Journal, 2023)

---

## 📁 Directory Structure

```
Lightweight_IA_V_2/
│
├── README.md                           # Project overview and roadmap
├── paper_2023_CANShield.pdf           # Reference paper (IEEE IoT 2023)
├── CANSHIELD_COMPLETE_STEPS.md        # Complete implementation guide
│
├── config/                             # Configuration files
│   ├── syncan_config.yaml             # SynCAN dataset configuration
│   ├── road_config.yaml               # ROAD dataset configuration
│   ├── model_config.yaml              # Model architecture configuration
│   └── detection_config.yaml          # Detection thresholds configuration
│
├── data/                               # Data directory
│   ├── raw/                           # Raw CAN datasets
│   │   ├── syncan/                    # SynCAN dataset
│   │   │   ├── ambient/               # Normal traffic (training)
│   │   │   │   ├── train_1.csv
│   │   │   │   ├── train_2.csv
│   │   │   │   ├── train_3.csv
│   │   │   │   └── train_4.csv
│   │   │   └── attacks/               # Attack traffic (testing)
│   │   │       ├── test_flooding.csv
│   │   │       ├── test_suppress.csv
│   │   │       ├── test_plateau.csv
│   │   │       ├── test_continuous.csv
│   │   │       └── test_playback.csv
│   │   └── road/                      # ROAD dataset (optional)
│   │
│   ├── processed/                     # Preprocessed data
│   │   ├── queue_snapshots/           # FIFO queue snapshots
│   │   ├── views/                     # Multi-scale views
│   │   │   ├── T_1/                   # Sampling period 1
│   │   │   ├── T_5/                   # Sampling period 5
│   │   │   ├── T_10/                  # Sampling period 10
│   │   │   ├── T_20/                  # Sampling period 20
│   │   │   └── T_50/                  # Sampling period 50
│   │   └── normalized/                # Normalized data
│   │
│   ├── scalers/                       # Min-Max scalers
│   │   ├── syncan_scaler.pkl          # SynCAN scaler
│   │   └── min_max_values.csv         # Min/Max values per signal
│   │
│   └── thresholds/                    # Detection thresholds
│       ├── R_Loss_T1.csv              # Tier 1 thresholds for T=1
│       ├── R_Loss_T5.csv              # Tier 1 thresholds for T=5
│       ├── R_Time_T1.csv              # Tier 2 thresholds for T=1
│       ├── R_Time_T5.csv              # Tier 2 thresholds for T=5
│       ├── R_Signal_T1.csv            # Tier 3 threshold for T=1
│       ├── R_Signal_T5.csv            # Tier 3 threshold for T=5
│       └── R_Signal_ensemble.csv      # Ensemble threshold
│
├── src/                                # Source code
│   │
│   ├── initialization/                 # Initialization phase
│   │   ├── __init__.py
│   │   ├── signal_selector.py         # Critical signal selection
│   │   ├── correlation_analyzer.py    # Pearson correlation & clustering
│   │   └── signal_reorderer.py        # Hierarchical clustering & reordering
│   │
│   ├── preprocessing/                  # Phase 1: Data Preprocessing
│   │   ├── __init__.py
│   │   ├── can_decoder.py             # CAN message decoder (DBC/CAN-D)
│   │   ├── fifo_queue.py              # FIFO queue Q management
│   │   ├── view_builder.py            # Multi-scale view creation
│   │   └── normalizer.py              # Min-Max normalization
│   │
│   ├── models/                         # Phase 2: Model Architecture
│   │   ├── __init__.py
│   │   ├── autoencoder.py             # CNN Autoencoder architecture
│   │   ├── layers.py                  # Custom layers (if needed)
│   │   └── transfer_learning.py       # Transfer learning utilities
│   │
│   ├── training/                       # Phase 2: Training Module
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop & callbacks
│   │   ├── threshold_generator.py     # Three-tier threshold generation
│   │   └── utils.py                   # Training utilities
│   │
│   ├── detection/                      # Phase 3: Detection Module
│   │   ├── __init__.py
│   │   ├── detector.py                # Main detection engine
│   │   ├── tier1_detector.py          # Pixel-level detection
│   │   ├── tier2_detector.py          # Signal-level detection
│   │   ├── tier3_detector.py          # Window-level scoring
│   │   └── ensemble.py                # Ensemble decision logic
│   │
│   ├── evaluation/                     # Evaluation & Metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                 # AUPRC, F1, Precision, Recall
│   │   ├── visualizer.py              # PR curves, confusion matrix
│   │   └── performance.py             # Latency measurement
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── logger.py                  # Logging utilities
│       ├── io_utils.py                # File I/O operations
│       └── config_loader.py           # Configuration loader
│
├── scripts/                            # Execution scripts
│   ├── 00_download_dataset.sh         # Download SynCAN dataset
│   ├── 01_initialize.py               # Run initialization phase
│   ├── 02_preprocess_data.py          # Preprocess raw data
│   ├── 03_train_autoencoders.py       # Train all autoencoders
│   ├── 04_generate_thresholds.py      # Generate detection thresholds
│   ├── 05_test_detection.py           # Test on attack datasets
│   ├── 06_evaluate_results.py         # Calculate metrics & visualize
│   └── 07_deploy_realtime.py          # Real-time detection demo
│
├── notebooks/                          # Jupyter notebooks (exploration)
│   ├── 01_data_exploration.ipynb      # Explore CAN dataset
│   ├── 02_correlation_analysis.ipynb  # Analyze signal correlations
│   ├── 03_model_training.ipynb        # Train models interactively
│   ├── 04_threshold_tuning.ipynb      # Tune detection thresholds
│   └── 05_results_visualization.ipynb # Visualize results
│
├── models/                             # Saved models
│   ├── syncan/                        # SynCAN models
│   │   ├── AE_T1.h5                   # Autoencoder for T=1
│   │   ├── AE_T5.h5                   # Autoencoder for T=5
│   │   ├── AE_T10.h5                  # Autoencoder for T=10
│   │   ├── AE_T20.h5                  # Autoencoder for T=20
│   │   ├── AE_T50.h5                  # Autoencoder for T=50
│   │   └── checkpoints/               # Training checkpoints
│   └── road/                          # ROAD models (optional)
│
├── results/                            # Experimental results
│   ├── training/                      # Training results
│   │   ├── histories/                 # Training histories (JSON)
│   │   ├── loss_curves/               # Loss plots
│   │   └── reconstructions/           # Sample reconstructions
│   │
│   ├── detection/                     # Detection results
│   │   ├── predictions/               # Predictions per attack type
│   │   ├── anomaly_scores/            # Anomaly scores
│   │   └── decisions/                 # Final attack/benign decisions
│   │
│   ├── evaluation/                    # Evaluation metrics
│   │   ├── metrics.csv                # AUPRC, F1, TPR, FPR
│   │   ├── pr_curves/                 # Precision-Recall curves
│   │   ├── confusion_matrices/        # Confusion matrices
│   │   └── latency_analysis/          # Latency measurements
│   │
│   └── comparison/                    # Comparison with baselines
│       ├── canshield_vs_canet.csv
│       └── plots/
│
├── tests/                              # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py          # Test preprocessing module
│   ├── test_models.py                 # Test model architecture
│   ├── test_training.py               # Test training pipeline
│   ├── test_detection.py              # Test detection logic
│   └── test_evaluation.py             # Test evaluation metrics
│
├── docs/                               # Documentation
│   ├── architecture.md                # System architecture
│   ├── api_reference.md               # API documentation
│   ├── installation.md                # Installation guide
│   ├── usage.md                       # Usage examples
│   └── troubleshooting.md             # Common issues & solutions
│
├── deployment/                         # Deployment files
│   ├── embedded/                      # Embedded deployment
│   │   ├── tflite_models/             # TensorFlow Lite models
│   │   ├── quantized_models/          # Quantized models
│   │   └── optimization_configs/      # Optimization settings
│   │
│   ├── docker/                        # Docker deployment
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── requirements.txt
│   │
│   └── cloud/                         # Cloud deployment (optional)
│       └── aws_lambda/
│
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
├── setup.py                            # Package setup
├── .gitignore                          # Git ignore file
└── LICENSE                             # License file
```

---

## 📚 Documentation

### **CANSHIELD_COMPLETE_STEPS.md**
Complete step-by-step guide for implementing CANShield:
- **Initialization Phase:** Signal selection and correlation clustering
- **Phase 1:** Data preprocessing module (multi-scale views)
- **Phase 2:** Data analyzer module (CNN autoencoder training)
- **Phase 3:** Attack detection module (real-time deployment)

**Includes:**
- Exact architecture details (5 conv layers: 32→16→16→32→1)
- Hyperparameters (Adam lr=0.0002, batch=128, epochs=100)
- Three-tier threshold mechanism
- Transfer learning strategy
- Ensemble decision logic

---

## 🎯 Project Goals

1. **Implement CANShield** from scratch following paper specifications
2. **Achieve <10ms detection latency** for real-time capability
3. **Replicate paper results** on SynCAN dataset (AUPRC ~0.95)
4. **Create lightweight version** for embedded deployment
5. **Test on real CAN data** (if available)

---

## 🔑 Key Features

- **Multi-Scale Temporal Analysis:** 5 sampling periods (1, 5, 10, 20, 50)
- **CNN Autoencoders:** Learn normal CAN traffic patterns
- **Transfer Learning:** Reduce training cost across scales
- **Three-Tier Thresholding:** Progressive anomaly filtering
- **Ensemble Decision:** Average scores from all models
- **Real-Time:** <10ms detection latency

---

## 📊 Expected Performance

| Metric | Target (from paper) |
|--------|---------------------|
| AUPRC (Ensemble) | 0.952 |
| Flooding Detection | 0.997 |
| Suppress Detection | 0.985 |
| Plateau Detection | 0.961 |
| Continuous Detection | 0.870 |
| Playback Detection | 0.948 |
| Detection Latency | <10ms |

---

## 🚀 Implementation Roadmap

### **Phase 1: Setup** ✅
- [x] Create project directory
- [x] Copy reference paper
- [x] Document complete steps
- [ ] Setup development environment
- [ ] Install dependencies

### **Phase 2: Data Preparation**
- [ ] Download SynCAN dataset
- [ ] Implement signal selection
- [ ] Implement correlation clustering
- [ ] Implement signal reordering

### **Phase 3: Preprocessing**
- [ ] Implement FIFO queue
- [ ] Implement CAN message decoder
- [ ] Implement multi-scale view creation
- [ ] Implement normalization

### **Phase 4: Training**
- [ ] Implement CNN autoencoder architecture
- [ ] Implement training loop
- [ ] Implement transfer learning
- [ ] Implement threshold generation
- [ ] Train on SynCAN dataset

### **Phase 5: Detection**
- [ ] Implement three-tier detection
- [ ] Implement ensemble decision
- [ ] Test on attack datasets
- [ ] Measure latency
- [ ] Validate results

### **Phase 6: Optimization**
- [ ] Profile performance bottlenecks
- [ ] Optimize for embedded deployment
- [ ] Convert to TensorFlow Lite
- [ ] Test on resource-constrained hardware

---

## 🛠️ Dependencies

```bash
# Python 3.9+
tensorflow>=2.10
keras>=2.10
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
```

---

## 📖 References

**Paper:**  
Md Hasan Shahriar, Yang Xiao, Pablo Moriano, Wenjing Lou, Y. Thomas Hou.  
*"CANShield: Deep-Learning-Based Intrusion Detection Framework for Controller Area Networks at the Signal Level"*  
IEEE Internet of Things Journal, 2023.  
DOI: 10.1109/JIOT.2023.3303271

**Original GitHub:**  
https://github.com/shahriar0651/CANShield

**Dataset:**  
SynCAN: https://github.com/etas/SynCAN

---

## 📝 Notes

- This is **Version 2** - completely rewritten based on accurate paper understanding
- Version 1 (Lightweight_IA) had some inaccuracies - use this version instead
- All steps verified against paper and original code
- Focus on exact replication before optimization

---

## ✅ Status

**Current Status:** Documentation Complete, Ready for Implementation  
**Next Step:** Setup environment and download dataset  
**Last Updated:** November 3, 2025

---

## 👤 Author

PhD Project - Automotive CAN Security  
Contact: abbad241@...

---
