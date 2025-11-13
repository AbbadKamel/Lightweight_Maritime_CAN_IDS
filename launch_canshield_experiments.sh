#!/bin/bash

# Launch CANShield-style hierarchical detection experiments in detached screen sessions
# Each experiment tests different factor combinations

echo "Launching CANShield-style hierarchical detection experiments..."
echo "================================================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Experiment 1: Balanced approach (95-95-95)
screen -dmS canshield_95_95_95 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo '========================================' | tee logs/canshield_95_95_95.log
    echo 'EXPERIMENT: CANShield Hierarchical Detection' | tee -a logs/canshield_95_95_95.log
    echo 'Loss Factor: p95 (per-signal threshold)' | tee -a logs/canshield_95_95_95.log
    echo 'Time Factor: p95 (temporal threshold)' | tee -a logs/canshield_95_95_95.log
    echo 'Signal Factor: p95 (window threshold)' | tee -a logs/canshield_95_95_95.log
    echo 'Started: \$(date)' | tee -a logs/canshield_95_95_95.log
    echo '========================================' | tee -a logs/canshield_95_95_95.log
    python3 run_detection_canshield_style.py 95 95 95 canshield_95_95_95 2>&1 | tee -a logs/canshield_95_95_95.log
    echo 'Completed: \$(date)' | tee -a logs/canshield_95_95_95.log
    exec bash
"

sleep 2

# Experiment 2: Strict approach (99-99-99)
screen -dmS canshield_99_99_99 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo '========================================' | tee logs/canshield_99_99_99.log
    echo 'EXPERIMENT: CANShield Hierarchical Detection' | tee -a logs/canshield_99_99_99.log
    echo 'Loss Factor: p99 (per-signal threshold - STRICT)' | tee -a logs/canshield_99_99_99.log
    echo 'Time Factor: p99 (temporal threshold - STRICT)' | tee -a logs/canshield_99_99_99.log
    echo 'Signal Factor: p99 (window threshold - STRICT)' | tee -a logs/canshield_99_99_99.log
    echo 'Started: \$(date)' | tee -a logs/canshield_99_99_99.log
    echo '========================================' | tee -a logs/canshield_99_99_99.log
    python3 run_detection_canshield_style.py 99 99 99 canshield_99_99_99 2>&1 | tee -a logs/canshield_99_99_99.log
    echo 'Completed: \$(date)' | tee -a logs/canshield_99_99_99.log
    exec bash
"

sleep 2

# Experiment 3: Mixed approach (95-99-95) - Strict temporal, balanced signal
screen -dmS canshield_95_99_95 bash -c "
    cd /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS
    source venv/bin/activate
    echo '========================================' | tee logs/canshield_95_99_95.log
    echo 'EXPERIMENT: CANShield Hierarchical Detection' | tee -a logs/canshield_95_99_95.log
    echo 'Loss Factor: p95 (per-signal threshold)' | tee -a logs/canshield_95_99_95.log
    echo 'Time Factor: p99 (temporal threshold - STRICT for flooding)' | tee -a logs/canshield_95_99_95.log
    echo 'Signal Factor: p95 (window threshold)' | tee -a logs/canshield_95_99_95.log
    echo 'Started: \$(date)' | tee -a logs/canshield_95_99_95.log
    echo '========================================' | tee -a logs/canshield_95_99_95.log
    python3 run_detection_canshield_style.py 95 99 95 canshield_95_99_95 2>&1 | tee -a logs/canshield_95_99_95.log
    echo 'Completed: \$(date)' | tee -a logs/canshield_95_99_95.log
    exec bash
"

sleep 2

echo ""
echo "All CANShield experiments launched!"
echo ""
echo "Active screen sessions:"
screen -ls | grep canshield

echo ""
echo "To monitor a specific experiment, use:"
echo "  ssh abbad241@N315L-G17G01.ressource.unicaen.fr 'screen -r canshield_95_95_95'"
echo ""
echo "To check logs:"
echo "  ssh abbad241@N315L-G17G01.ressource.unicaen.fr 'tail -f /data/chercheurs/abbad241/Lightweight_Maritime_CAN_IDS/logs/canshield_95_95_95.log'"
