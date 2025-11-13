#!/bin/bash
# Launch 4 parallel detection experiments on supercomputer
# Each runs in its own screen session

echo "================================================================================"
echo "LAUNCHING 4 PARALLEL DETECTION EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Experiments:"
echo "  1. vote2_p95 - Voting=2, p95 threshold (your best local config)"
echo "  2. vote3_p95 - Voting=3, p95 threshold (current baseline)"
echo "  3. vote2_p99 - Voting=2, p99 threshold (strict)"
echo "  4. vote1_p95 - Voting=1, p95 threshold (most sensitive)"
echo ""
echo "Each experiment will:"
echo "  - Run in separate screen session"
echo "  - Process full 4.9M samples"
echo "  - Save results to separate files"
echo "  - Take ~20-30 minutes"
echo ""
echo "================================================================================"
echo ""

cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS

# Activate virtual environment
source venv/bin/activate

# Experiment 1: vote2_p95 (best from local)
echo "Starting Experiment 1: vote2_p95..."
screen -dmS exp_vote2_p95 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo 'Experiment vote2_p95 started at \$(date)' > logs/exp_vote2_p95.log
    python3 run_detection_experiment.py 2 p95 vote2_p95 >> logs/exp_vote2_p95.log 2>&1
    echo 'Experiment vote2_p95 completed at \$(date)' >> logs/exp_vote2_p95.log
    exec bash
"
echo "✓ Launched: exp_vote2_p95"

# Experiment 2: vote3_p95 (current baseline)
echo "Starting Experiment 2: vote3_p95..."
screen -dmS exp_vote3_p95 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo 'Experiment vote3_p95 started at \$(date)' > logs/exp_vote3_p95.log
    python3 run_detection_experiment.py 3 p95 vote3_p95 >> logs/exp_vote3_p95.log 2>&1
    echo 'Experiment vote3_p95 completed at \$(date)' >> logs/exp_vote3_p95.log
    exec bash
"
echo "✓ Launched: exp_vote3_p95"

# Experiment 3: vote2_p99 (strict threshold)
echo "Starting Experiment 3: vote2_p99..."
screen -dmS exp_vote2_p99 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo 'Experiment vote2_p99 started at \$(date)' > logs/exp_vote2_p99.log
    python3 run_detection_experiment.py 2 p99 vote2_p99 >> logs/exp_vote2_p99.log 2>&1
    echo 'Experiment vote2_p99 completed at \$(date)' >> logs/exp_vote2_p99.log
    exec bash
"
echo "✓ Launched: exp_vote2_p99"

# Experiment 4: vote1_p95 (most sensitive)
echo "Starting Experiment 4: vote1_p95..."
screen -dmS exp_vote1_p95 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo 'Experiment vote1_p95 started at \$(date)' > logs/exp_vote1_p95.log
    python3 run_detection_experiment.py 1 p95 vote1_p95 >> logs/exp_vote1_p95.log 2>&1
    echo 'Experiment vote1_p95 completed at \$(date)' >> logs/exp_vote1_p95.log
    exec bash
"
echo "✓ Launched: exp_vote1_p95"

echo ""
echo "================================================================================"
echo "ALL 4 EXPERIMENTS LAUNCHED!"
echo "================================================================================"
echo ""
echo "Check status with:"
echo "  screen -ls"
echo ""
echo "Attach to specific experiment:"
echo "  screen -r exp_vote2_p95"
echo "  screen -r exp_vote3_p95"
echo "  screen -r exp_vote2_p99"
echo "  screen -r exp_vote1_p95"
echo ""
echo "Detach with: Ctrl-a d"
echo ""
echo "View logs:"
echo "  tail -f logs/exp_vote2_p95.log"
echo ""
echo "Expected completion: ~20-30 minutes"
echo "================================================================================"
