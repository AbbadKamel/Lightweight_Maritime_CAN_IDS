#!/bin/bash

# Test the CANShield chunked implementation with small subset first

echo "================================================================================"
echo "CANSHIELD CHUNKED DETECTION - TEST SUITE"
echo "================================================================================"
echo ""
echo "This script will test the CANShield implementation with increasing data sizes"
echo "to ensure it works correctly before running on the full 4.9M dataset."
echo ""
echo "================================================================================"
echo ""

# Test 1: Quick validation (100K timesteps)
echo "TEST 1: Quick Validation (100,000 timesteps)"
echo "=============================================="
echo "Purpose: Verify code works, no crashes, fast execution"
echo "Expected: ~1-2 minutes"
echo "Data: ~6K samples per attack type, ~70K normal"
echo ""
echo "Starting..."
python3 run_canshield_chunked.py 100000 95 95 95 test_100k_95_95_95

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TEST 1 PASSED - Code works!"
    echo ""
    
    # Check results
    if [ -f "results/detection/summary_test_100k_95_95_95.json" ]; then
        echo "Results saved. Checking metrics..."
        python3 -c "
import json
with open('results/detection/summary_test_100k_95_95_95.json') as f:
    r = json.load(f)
print(f\"  Precision: {r['metrics']['precision']*100:.2f}%\")
print(f\"  Recall:    {r['metrics']['recall']*100:.2f}%\")
print(f\"  F1-Score:  {r['metrics']['f1_score']*100:.2f}%\")

# Warnings
if r['metrics']['precision'] < 0.10:
    print('  ⚠️  WARNING: Very low precision - may be flagging too much')
elif r['metrics']['precision'] > 0.99:
    print('  ⚠️  WARNING: Very high precision but check recall!')
    
if r['metrics']['recall'] < 0.10:
    print('  ⚠️  WARNING: Very low recall - missing most attacks')
elif r['metrics']['recall'] > 0.99:
    print('  ⚠️  WARNING: Very high recall - check if flagging everything')
    
if r['metrics']['f1_score'] > 0.80:
    print('  ✅ EXCELLENT F1-Score - Method is working well!')
elif r['metrics']['f1_score'] > 0.60:
    print('  ✅ GOOD F1-Score - Method shows promise')
elif r['metrics']['f1_score'] > 0.40:
    print('  ⚠️  MODERATE F1-Score - May need tuning')
else:
    print('  ❌ LOW F1-Score - Method needs adjustment')
"
    fi
    
    echo ""
    echo "================================================================================"
    echo ""
    read -p "Continue to TEST 2 (500K timesteps)? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "TEST 2: Validation Run (500,000 timesteps)"
        echo "==========================================="
        echo "Purpose: Get reliable thresholds, validate detection quality"
        echo "Expected: ~5-10 minutes"
        echo "Data: ~30K samples per attack type, ~350K normal"
        echo ""
        echo "Starting..."
        python3 run_canshield_chunked.py 500000 95 95 95 test_500k_95_95_95
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ TEST 2 PASSED - Validation successful!"
            echo ""
            
            # Compare results
            echo "Comparing 100K vs 500K results..."
            python3 -c "
import json

with open('results/detection/summary_test_100k_95_95_95.json') as f:
    r1 = json.load(f)
with open('results/detection/summary_test_500k_95_95_95.json') as f:
    r2 = json.load(f)

print(f\"  {'Metric':<15} {'100K':>10} {'500K':>10} {'Change':>10}\")
print(f\"  {'-'*50}\")

metrics = ['precision', 'recall', 'f1_score']
for m in metrics:
    v1 = r1['metrics'][m] * 100
    v2 = r2['metrics'][m] * 100
    diff = v2 - v1
    sign = '+' if diff > 0 else ''
    print(f\"  {m.capitalize():<15} {v1:>9.2f}% {v2:>9.2f}% {sign}{diff:>8.2f}%\")

print()
if abs(r1['metrics']['f1_score'] - r2['metrics']['f1_score']) < 0.05:
    print('  ✅ Results are STABLE across dataset sizes')
else:
    print('  ⚠️  Results vary significantly - may need more data')
"
            
            echo ""
            echo "================================================================================"
            echo ""
            echo "✅ ALL TESTS PASSED!"
            echo ""
            echo "Next step: Run on FULL dataset (4.9M timesteps)"
            echo "Command: python3 run_canshield_chunked.py -1 95 95 95 full_95_95_95"
            echo ""
            echo "Or try different factor combinations:"
            echo "  - Strict:  python3 run_canshield_chunked.py -1 99 99 99 full_99_99_99"
            echo "  - Mixed:   python3 run_canshield_chunked.py -1 95 99 95 full_95_99_95"
            echo ""
        else
            echo "❌ TEST 2 FAILED"
            exit 1
        fi
    else
        echo ""
        echo "Test 2 skipped. You can run it manually:"
        echo "  python3 run_canshield_chunked.py 500000 95 95 95 test_500k_95_95_95"
    fi
else
    echo ""
    echo "❌ TEST 1 FAILED - Check error messages above"
    exit 1
fi

echo "================================================================================"
