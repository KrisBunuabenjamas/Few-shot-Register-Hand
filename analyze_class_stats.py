#!/usr/bin/env python3
"""Analyze per-class statistics from evaluation results"""

import json
import argparse
from collections import defaultdict


def analyze_class_stats(results_json, output_json):
    # Load results
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    # Per-class statistics
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'failed': 0})
    
    for result in data['results']:
        true_class = result['true_class']
        is_correct = result['correct']
        
        class_stats[true_class]['total'] += 1
        if is_correct:
            class_stats[true_class]['correct'] += 1
        else:
            class_stats[true_class]['failed'] += 1
    
    # Calculate accuracy per class
    for cls in class_stats:
        total = class_stats[cls]['total']
        correct = class_stats[cls]['correct']
        class_stats[cls]['accuracy'] = correct / total if total > 0 else 0
    
    # Sort by class name
    sorted_stats = {k: class_stats[k] for k in sorted(class_stats.keys())}
    
    # Overall statistics
    overall = {
        'total_classes': len(class_stats),
        'total_samples': data['total'],
        'total_correct': data['correct'],
        'total_failed': data['total'] - data['correct'],
        'overall_accuracy': data['accuracy_all'],
        'per_class_stats': sorted_stats
    }
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(overall, f, indent=2)
    
    # Print summary
    print("=" * 70)
    print(f"{'Class':<10} {'Total':<10} {'Correct':<10} {'Failed':<10} {'Accuracy':<10}")
    print("=" * 70)
    
    for cls, stats in sorted_stats.items():
        print(f"{cls:<10} {stats['total']:<10} {stats['correct']:<10} {stats['failed']:<10} {stats['accuracy']:.2%}")
    
    print("=" * 70)
    print(f"{'OVERALL':<10} {overall['total_samples']:<10} {overall['total_correct']:<10} {overall['total_failed']:<10} {overall['overall_accuracy']:.2%}")
    print("=" * 70)
    
    print(f"\nSaved detailed statistics to: {output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze per-class statistics')
    parser.add_argument('--results', default='evaluation_results.json', help='Evaluation results JSON')
    parser.add_argument('--out', default='class_statistics.json', help='Output statistics JSON')
    
    args = parser.parse_args()
    
    analyze_class_stats(args.results, args.out)
