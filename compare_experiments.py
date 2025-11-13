#!/usr/bin/env python3
"""
Compare Results from All Detection Experiments
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
DETECTION_DIR = BASE_DIR / 'results' / 'detection'

# Experiments to compare
EXPERIMENTS = [
    'vote1_p95',
    'vote2_p95',
    'vote2_p99',
    'vote3_p95'
]

print("="*100)
print("COMPARISON OF ALL DETECTION EXPERIMENTS")
print("="*100)
print()

# Load all results
results = {}
for exp in EXPERIMENTS:
    summary_file = DETECTION_DIR / f'summary_{exp}.json'
    if summary_file.exists():
        with open(summary_file) as f:
            results[exp] = json.load(f)
    else:
        print(f"⚠️  Missing: {exp}")

if not results:
    print("❌ No results found!")
    exit(1)

print(f"Found {len(results)} completed experiments")
print()

# ============================================================================
# Overall Metrics Comparison
# ============================================================================

print("="*100)
print("OVERALL METRICS COMPARISON")
print("="*100)
print()
print(f"{'Experiment':<15} {'Voting':<10} {'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-"*100)

best_f1 = 0
best_exp = None

for exp in EXPERIMENTS:
    if exp not in results:
        continue
    
    r = results[exp]
    config = r['configuration']
    metrics = r['metrics']
    
    voting = f"{config['voting_threshold']}/5"
    threshold = config['threshold_percentile']
    
    acc = metrics['accuracy'] * 100
    prec = metrics['precision'] * 100
    rec = metrics['recall'] * 100
    f1 = metrics['f1_score'] * 100
    
    # Track best F1
    if f1 > best_f1:
        best_f1 = f1
        best_exp = exp
    
    print(f"{exp:<15} {voting:<10} {threshold:<12} {acc:>10.2f}% {prec:>10.2f}% {rec:>10.2f}% {f1:>10.2f}%")

print()
print(f"🏆 BEST F1 SCORE: {best_exp} ({best_f1:.2f}%)")
print()

# ============================================================================
# Per-Attack Detection Rates
# ============================================================================

print("="*100)
print("PER-ATTACK DETECTION RATES")
print("="*100)
print()

# Get attack types from first result
attack_types = list(next(iter(results.values()))['per_attack_stats'].keys())

for attack in attack_types:
    print(f"\n{attack}:")
    print(f"{'Experiment':<15} {'Detected':<15} {'Total':<12} {'Rate':<12}")
    print("-"*60)
    
    for exp in EXPERIMENTS:
        if exp not in results:
            continue
        
        stats = results[exp]['per_attack_stats'][attack]
        detected = stats['detected']
        total = stats['total']
        rate = stats['rate'] * 100
        
        print(f"{exp:<15} {detected:>7,}/{total:<7,} {rate:>10.2f}%")

print()

# ============================================================================
# Confusion Matrix Comparison
# ============================================================================

print("="*100)
print("CONFUSION MATRICES")
print("="*100)
print()

for exp in EXPERIMENTS:
    if exp not in results:
        continue
    
    cm = results[exp]['confusion_matrix']
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    print(f"{exp}:")
    print(f"                 Predicted")
    print(f"               Normal  Attack")
    print(f"  Actual Normal  {tn:>7,} {fp:>7,}")
    print(f"         Attack  {fn:>7,} {tp:>7,}")
    print()

# ============================================================================
# Trade-off Analysis
# ============================================================================

print("="*100)
print("PRECISION-RECALL TRADE-OFF ANALYSIS")
print("="*100)
print()

print(f"{'Experiment':<15} {'False Positives':<18} {'False Negatives':<18} {'FP Rate':<12} {'FN Rate':<12}")
print("-"*100)

for exp in EXPERIMENTS:
    if exp not in results:
        continue
    
    cm = results[exp]['confusion_matrix']
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    total_normal = tn + fp
    total_attack = fn + tp
    
    fp_rate = (fp / total_normal * 100) if total_normal > 0 else 0
    fn_rate = (fn / total_attack * 100) if total_attack > 0 else 0
    
    print(f"{exp:<15} {fp:>9,}/{total_normal:<7,} {fn:>9,}/{total_attack:<7,} {fp_rate:>10.2f}% {fn_rate:>10.2f}%")

print()

# ============================================================================
# Recommendations
# ============================================================================

print("="*100)
print("RECOMMENDATIONS")
print("="*100)
print()

# Find configuration with best balance
best_balanced = None
best_balanced_score = 0

for exp in EXPERIMENTS:
    if exp not in results:
        continue
    
    metrics = results[exp]['metrics']
    # Balanced score: harmonic mean of precision and recall (F1)
    score = metrics['f1_score']
    
    if score > best_balanced_score:
        best_balanced_score = score
        best_balanced = exp

# Find configuration with highest recall
best_recall = None
best_recall_score = 0

for exp in EXPERIMENTS:
    if exp not in results:
        continue
    
    metrics = results[exp]['metrics']
    if metrics['recall'] > best_recall_score:
        best_recall_score = metrics['recall']
        best_recall = exp

# Find configuration with highest precision
best_precision = None
best_precision_score = 0

for exp in EXPERIMENTS:
    if exp not in results:
        continue
    
    metrics = results[exp]['metrics']
    if metrics['precision'] > best_precision_score:
        best_precision_score = metrics['precision']
        best_precision = exp

print(f"🎯 Best Balance (F1):       {best_balanced} (F1={best_balanced_score*100:.2f}%)")
print(f"🔍 Best Recall (Catch all): {best_recall} (Recall={best_recall_score*100:.2f}%)")
print(f"✅ Best Precision (Low FP): {best_precision} (Precision={best_precision_score*100:.2f}%)")
print()

print("For maritime safety (critical application):")
print("  → Recommend: Configuration with HIGHEST RECALL")
print(f"  → Use: {best_recall}")
print("  → Rationale: Better to have false alarms than miss real attacks")
print()

print("For production deployment (minimize alerts):")
print("  → Recommend: Configuration with BEST F1 SCORE")
print(f"  → Use: {best_balanced}")
print("  → Rationale: Best balance between catching attacks and avoiding false alarms")
print()

print("="*100)

# Save comparison report
report_file = DETECTION_DIR / 'comparison_report.json'
comparison_data = {
    'timestamp': datetime.now().isoformat(),
    'experiments': results,
    'recommendations': {
        'best_balanced': {
            'experiment': best_balanced,
            'f1_score': float(best_balanced_score)
        },
        'best_recall': {
            'experiment': best_recall,
            'recall': float(best_recall_score)
        },
        'best_precision': {
            'experiment': best_precision,
            'precision': float(best_precision_score)
        }
    }
}

with open(report_file, 'w') as f:
    json.dump(comparison_data, f, indent=2)

print(f"Full comparison saved to: {report_file}")
print()
