#!/bin/bash
# Launch 4 parallel detection experiments on supercomputer
# Each runs in its own detached screen session with live output

echo "================================================================================"
echo "LAUNCHING 4 PARALLEL DETECTION EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Experiments:"
echo "  1. vote2_p95 - Voting=2/5, p95 threshold (YOUR BEST LOCAL CONFIG)"
echo "  2. vote3_p95 - Voting=3/5, p95 threshold (CURRENT BASELINE)"
echo "  3. vote2_p99 - Voting=2/5, p99 threshold (STRICT - Low FP)"
echo "  4. vote1_p95 - Voting=1/5, p95 threshold (MOST SENSITIVE - High Recall)"
echo ""
echo "Each experiment will:"
echo "  - Run in DETACHED screen session"
echo "  - Process FULL 4.9M samples"
echo "  - Save results to separate files"
echo "  - Take ~20-30 minutes"
echo ""
echo "================================================================================"
echo ""

# Change to project directory
cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS || exit 1

# Create logs directory
mkdir -p logs

# Experiment 1: vote2_p95 (best from local - expected 98% recall)
echo "[1/4] Launching vote2_p95 (Voting=2/5, p95) - Expected: High Recall ~98%"
screen -dmS exp_vote2_p95 bash -c '
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo "========================================" | tee logs/exp_vote2_p95.log
    echo "EXPERIMENT: vote2_p95" | tee -a logs/exp_vote2_p95.log
    echo "Voting: 2/5 autoencoders" | tee -a logs/exp_vote2_p95.log
    echo "Threshold: p95" | tee -a logs/exp_vote2_p95.log
    echo "Started: $(date)" | tee -a logs/exp_vote2_p95.log
    echo "========================================" | tee -a logs/exp_vote2_p95.log
    python3 run_detection_experiment.py 2 p95 vote2_p95 2>&1 | tee -a logs/exp_vote2_p95.log
    echo "Completed: $(date)" | tee -a logs/exp_vote2_p95.log
    exec bash
'
echo "  ✓ Screen: exp_vote2_p95 (detached)"
echo "  ✓ Log: logs/exp_vote2_p95.log"
echo ""

# Experiment 2: vote3_p95 (current baseline - expected 21% recall)
echo "[2/4] Launching vote3_p95 (Voting=3/5, p95) - Expected: Low Recall ~21%"
screen -dmS exp_vote3_p95 bash -c '
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo "========================================" | tee logs/exp_vote3_p95.log
    echo "EXPERIMENT: vote3_p95" | tee -a logs/exp_vote3_p95.log
    echo "Voting: 3/5 autoencoders" | tee -a logs/exp_vote3_p95.log
    echo "Threshold: p95" | tee -a logs/exp_vote3_p95.log
    echo "Started: $(date)" | tee -a logs/exp_vote3_p95.log
    echo "========================================" | tee -a logs/exp_vote3_p95.log
    python3 run_detection_experiment.py 3 p95 vote3_p95 2>&1 | tee -a logs/exp_vote3_p95.log
    echo "Completed: $(date)" | tee -a logs/exp_vote3_p95.log
    exec bash
'
echo "  ✓ Screen: exp_vote3_p95 (detached)"
echo "  ✓ Log: logs/exp_vote3_p95.log"
echo ""

# Experiment 3: vote2_p99 (strict threshold - expected low FP, low recall)
echo "[3/4] Launching vote2_p99 (Voting=2/5, p99) - Expected: Low FP, Low Recall"
screen -dmS exp_vote2_p99 bash -c '
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo "========================================" | tee logs/exp_vote2_p99.log
    echo "EXPERIMENT: vote2_p99" | tee -a logs/exp_vote2_p99.log
    echo "Voting: 2/5 autoencoders" | tee -a logs/exp_vote2_p99.log
    echo "Threshold: p99 (STRICT)" | tee -a logs/exp_vote2_p99.log
    echo "Started: $(date)" | tee -a logs/exp_vote2_p99.log
    echo "========================================" | tee -a logs/exp_vote2_p99.log
    python3 run_detection_experiment.py 2 p99 vote2_p99 2>&1 | tee -a logs/exp_vote2_p99.log
    echo "Completed: $(date)" | tee -a logs/exp_vote2_p99.log
    exec bash
'
echo "  ✓ Screen: exp_vote2_p99 (detached)"
echo "  ✓ Log: logs/exp_vote2_p99.log"
echo ""

# Experiment 4: vote1_p95 (most sensitive - expected very high recall, high FP)
echo "[4/4] Launching vote1_p95 (Voting=1/5, p95) - Expected: Very High Recall, High FP"
screen -dmS exp_vote1_p95 bash -c '
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo "========================================" | tee logs/exp_vote1_p95.log
    echo "EXPERIMENT: vote1_p95" | tee -a logs/exp_vote1_p95.log
    echo "Voting: 1/5 autoencoders (MOST SENSITIVE)" | tee -a logs/exp_vote1_p95.log
    echo "Threshold: p95" | tee -a logs/exp_vote1_p95.log
    echo "Started: $(date)" | tee -a logs/exp_vote1_p95.log
    echo "========================================" | tee -a logs/exp_vote1_p95.log
    python3 run_detection_experiment.py 1 p95 vote1_p95 2>&1 | tee -a logs/exp_vote1_p95.log
    echo "Completed: $(date)" | tee -a logs/exp_vote1_p95.log
    exec bash
'
echo "  ✓ Screen: exp_vote1_p95 (detached)"
echo "  ✓ Log: logs/exp_vote1_p95.log"
echo ""

sleep 2
echo "================================================================================"
echo "✅ ALL 4 EXPERIMENTS LAUNCHED IN DETACHED SCREENS!"
echo "================================================================================"
echo ""
echo "Active screen sessions:"
screen -ls | grep exp_
echo ""
echo "📊 Monitor progress:"
echo "  tail -f logs/exp_vote2_p95.log    # Best local config"
echo "  tail -f logs/exp_vote3_p95.log    # Current baseline"
echo "  tail -f logs/exp_vote2_p99.log    # Strict threshold"
echo "  tail -f logs/exp_vote1_p95.log    # Most sensitive"
echo ""
echo "🔗 Attach to live session:"
echo "  screen -r exp_vote2_p95           # Then Ctrl-a d to detach"
echo "  screen -r exp_vote3_p95"
echo "  screen -r exp_vote2_p99"
echo "  screen -r exp_vote1_p95"
echo ""
echo "📈 Check if experiments are running:"
echo "  ps aux | grep run_detection_experiment.py"
echo ""
echo "⏱️  Expected completion: ~20-30 minutes (around $(date -d '+25 minutes' '+%H:%M'))"
echo ""
echo "🎯 When complete, compare results:"
echo "  python3 compare_experiments.py"
echo ""
echo "================================================================================"
