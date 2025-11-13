#!/bin/bash

# Monitor all running experiments (old MSE voting + new CANShield hierarchical)

echo "=========================================="
echo "EXPERIMENT STATUS CHECK"
echo "Current time: $(date)"
echo "=========================================="
echo ""

# Check if we're running on the supercomputer or local machine
if [[ $(hostname) == *"N315L-G17G01"* ]]; then
    # Running on supercomputer
    IS_REMOTE=false
else
    # Running locally, need SSH
    IS_REMOTE=true
fi

function run_cmd() {
    if $IS_REMOTE; then
        ssh abbad241@N315L-G17G01.ressource.unicaen.fr "$1"
    else
        eval "$1"
    fi
}

echo "=== SCREEN SESSIONS ==="
run_cmd 'screen -ls | grep -E "(exp_vote|canshield)" || echo "No experiments running"'
echo ""

echo "=== PYTHON PROCESSES (Detection Experiments) ==="
run_cmd 'ps aux | grep "python3 run_detection" | grep -v grep | wc -l' | xargs echo "Active processes:"
echo ""

echo "=== OLD APPROACH (MSE Voting) - Started 11:21 ==="
run_cmd 'ps aux | grep "run_detection_experiment.py" | grep python3 | grep -v grep | awk "{printf \"%-15s CPU: %5s%% RAM: %4s%% Args: %s %s %s\n\", \$12, \$3, \$4, \$13, \$14, \$15}"'
echo ""

echo "=== NEW APPROACH (CANShield Hierarchical) - Started 11:42 ==="
run_cmd 'ps aux | grep "run_detection_canshield_style.py" | grep python3 | grep -v grep | awk "{printf \"%-25s CPU: %5s%% RAM: %4s%% Args: %s %s %s\n\", \$12, \$3, \$4, \$13, \$14, \$15}"'
echo ""

echo "=== RESULTS FILES ==="
run_cmd 'cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS && ls -lh results_*.json 2>/dev/null | awk "{print \$9, \$5}" || echo "No results yet"'
echo ""

echo "=== LATEST LOG ENTRIES ==="
echo ""
echo "--- Old Approach (vote2_p95 - expected best) ---"
run_cmd 'tail -3 /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS/logs/exp_vote2_p95.log 2>/dev/null || echo "Log not available"'
echo ""
echo "--- New Approach (canshield_95_95_95 - balanced) ---"
run_cmd 'tail -3 /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS/logs/canshield_95_95_95.log 2>/dev/null || echo "Log not available"'
echo ""

echo "=========================================="
echo "To monitor a specific experiment:"
echo "  screen -r exp_vote2_p95"
echo "  screen -r canshield_95_95_95"
echo ""
echo "To check full log:"
echo "  tail -f logs/exp_vote2_p95.log"
echo "  tail -f logs/canshield_95_95_95.log"
echo "=========================================="
