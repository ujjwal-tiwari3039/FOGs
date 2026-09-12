#!/usr/bin/env python3
"""
local_eval.py — ED-05 Test Set Evaluation (FIREWALL)
=====================================================
This is the ONLY script allowed to touch test/3way/**.
Its ONLY output is printed numbers — nothing feeds back into
training, features, or thresholds.

Usage:
    python local_eval.py [--use-nli]
"""

import sys
import random
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from src.load_data import load_training_data, load_test_data, print_label_stats
from src.features import init_models, cache_reference_embeddings
from src.model import load_model, predict
from src.metrics import macro_f1, guardrail_check, CLASS_ORDER

from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser(description="ED-05 Test Set Evaluation")
    parser.add_argument('--use-nli', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("  ED-05 LOCAL EVALUATION — Test Set Generalization")
    print("  ⚠️  This script reads test/3way/** for reporting ONLY.")
    print("  ⚠️  No output feeds back into training or features.")
    print("=" * 60)

    # Load training data (for reference embedding cache only)
    train_data = load_training_data()

    # Load test data
    test_data = load_test_data()
    print_label_stats(test_data, "Test Set (all)")

    # Initialize models and cache
    init_models(use_nli=args.use_nli)
    cache_reference_embeddings(train_data + test_data)

    # Load trained model
    model = load_model()

    # Overall evaluation
    test_preds, test_probs = predict(model, test_data, use_nli=args.use_nli)
    test_true = [d['label'] for d in test_data]

    f1 = macro_f1(test_true, test_preds)
    print(f"\n  Overall Test Macro-F1: {f1:.4f}")
    print(f"\n{classification_report(test_true, test_preds, target_names=CLASS_ORDER)}")

    # Per-condition evaluation
    conditions = sorted(set(d.get('condition', 'unknown') for d in test_data))
    print("\n" + "─" * 60)
    print("  Per-Condition Macro-F1:")
    print("─" * 60)

    condition_results = {}
    for cond in conditions:
        cond_data = [d for d in test_data if d.get('condition') == cond]
        cond_preds = [test_preds[i] for i, d in enumerate(test_data)
                      if d.get('condition') == cond]
        cond_true = [d['label'] for d in cond_data]

        cond_f1 = macro_f1(cond_true, cond_preds)
        guardrail = guardrail_check(cond_preds, cond_true)
        condition_results[cond] = cond_f1

        print(f"  {cond:25s}: F1={cond_f1:.4f}  (n={len(cond_data)})")
        for w in guardrail['warnings']:
            print(f"    {w}")

    # Per-domain evaluation
    domains = sorted(set(d['domain'] for d in test_data))
    print("\n" + "─" * 60)
    print("  Per-Domain Macro-F1:")
    print("─" * 60)

    for domain in domains:
        dom_data = [d for d in test_data if d['domain'] == domain]
        dom_preds = [test_preds[i] for i, d in enumerate(test_data)
                     if d['domain'] == domain]
        dom_true = [d['label'] for d in dom_data]

        dom_f1 = macro_f1(dom_true, dom_preds)
        print(f"  {domain:25s}: F1={dom_f1:.4f}  (n={len(dom_data)})")

    print("\n" + "=" * 60)
    print("  ✅ Evaluation complete. No data fed back to training.")
    print("=" * 60)

    return condition_results


if __name__ == '__main__':
    main()
