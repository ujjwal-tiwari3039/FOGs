#!/usr/bin/env python3
"""
run_all.py — ED-05 Master Pipeline
===================================
Single command to: train, evaluate, generate counterfactuals, print scorecard,
and write example explanations. This is the main entry point.

Usage:
    python run_all.py [--use-nli] [--skip-train]
"""

import sys
import os
import random
import argparse
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Fix ALL random seeds before any imports that might use them
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
except ImportError:
    pass

from src.load_data import load_training_data, make_train_val_split, print_label_stats
from src.preprocess import preprocess
from src.counterfactuals import cf1, cf2, cf3, cf4, fit_cf4_vectorizer, generate_all_counterfactuals
from src.features import init_models, cache_reference_embeddings, extract_features_batch
from src.model import train_model, predict, save_model, load_model
from src.metrics import (
    macro_f1, counterfactual_consistency, cf4_resistance,
    normalized_brier_score, guardrail_check, print_scorecard, CLASS_ORDER
)
from src.explain import generate_example_explanations


def main():
    parser = argparse.ArgumentParser(description="ED-05 Full Pipeline")
    parser.add_argument('--use-nli', action='store_true',
                        help='Use NLI cross-encoder features (stretch)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Load saved model instead of training')
    args = parser.parse_args()

    use_nli = args.use_nli

    print("=" * 60)
    print("  ED-05: Bias-Resistant Short-Answer Assessment")
    print("  Full Pipeline Run")
    print("=" * 60)

    # ──────────────────────────────────────────
    # STEP 1: Load and split data
    # ──────────────────────────────────────────
    print("\n📦 STEP 1: Loading training data...")
    train_data_full = load_training_data()
    print_label_stats(train_data_full, "Full Training Set")

    train_data, val_data = make_train_val_split(train_data_full)
    print_label_stats(train_data, "Train Split")
    print_label_stats(val_data, "Validation Split")

    # ──────────────────────────────────────────
    # STEP 2: Train or load model
    # ──────────────────────────────────────────
    if args.skip_train and (Path(__file__).parent / "models" / "model.pkl").exists():
        print("\n🔧 STEP 2: Loading saved model...")
        init_models(use_nli=use_nli)
        cache_reference_embeddings(train_data_full)
        model = load_model()
    else:
        print("\n🔧 STEP 2: Training model...")
        model = train_model(train_data, val_data, use_nli=use_nli, calibrate=True)
        save_model(model)

    # ──────────────────────────────────────────
    # STEP 3: Evaluate on validation set
    # ──────────────────────────────────────────
    print("\n📊 STEP 3: Evaluating on validation set...")
    val_preds, val_probs = predict(model, val_data, use_nli=use_nli)
    val_true = [d['label'] for d in val_data]

    f1 = macro_f1(val_true, val_preds)
    print(f"  Macro-F1: {f1:.4f}")

    # ──────────────────────────────────────────
    # STEP 4: Counterfactual evaluation
    # ──────────────────────────────────────────
    print("\n🔄 STEP 4: Counterfactual evaluation...")

    # Fit CF4 vectorizer on training answers only
    training_answers = [d['student_answer'] for d in train_data]
    cf4_vectorizer = fit_cf4_vectorizer(training_answers)

    # Generate CF variants and predict on them
    cf_preds_list = []  # For consistency: list of [cf1_pred, cf2_pred, cf3_pred]
    p_correct_original = []
    p_correct_cf4 = []

    correct_idx = CLASS_ORDER.index('correct')

    print("  Generating counterfactual predictions...")
    batch_size = 100
    for batch_start in range(0, len(val_data), batch_size):
        batch_end = min(batch_start + batch_size, len(val_data))
        batch = val_data[batch_start:batch_end]

        # Original predictions (already computed)
        for i, d in enumerate(batch):
            global_idx = batch_start + i
            p_correct_original.append(float(val_probs[global_idx][correct_idx]))

        # CF1 variants
        cf1_data = []
        cf2_data = []
        cf3_data = []
        cf4_data = []

        for d in batch:
            for cf_func, cf_list in [(cf1, cf1_data), (cf2, cf2_data),
                                      (cf3, cf3_data)]:
                cf_record = dict(d)
                cf_record['student_answer'] = cf_func(d['student_answer'])
                cf_list.append(cf_record)

            cf4_record = dict(d)
            cf4_record['student_answer'] = cf4(d['student_answer'], cf4_vectorizer)
            cf4_data.append(cf4_record)

        cf1_preds, _ = predict(model, cf1_data, use_nli=use_nli)
        cf2_preds, _ = predict(model, cf2_data, use_nli=use_nli)
        cf3_preds, _ = predict(model, cf3_data, use_nli=use_nli)
        cf4_preds, cf4_probs = predict(model, cf4_data, use_nli=use_nli)

        for i in range(len(batch)):
            cf_preds_list.append([cf1_preds[i], cf2_preds[i], cf3_preds[i]])
            p_correct_cf4.append(float(cf4_probs[i][correct_idx]))

    # Compute metrics
    consistency = counterfactual_consistency(val_preds, cf_preds_list)
    cf4_util = cf4_resistance(p_correct_original, p_correct_cf4)
    brier = normalized_brier_score(val_true, val_probs)
    guardrail = guardrail_check(val_preds, val_true)

    # ──────────────────────────────────────────
    # STEP 5: Print scorecard
    # ──────────────────────────────────────────
    print("\n📋 STEP 5: Scorecard")
    total = print_scorecard(f1, consistency, cf4_util, brier, guardrail)

    # ──────────────────────────────────────────
    # STEP 6: Explainability outputs
    # ──────────────────────────────────────────
    print("\n📝 STEP 6: Generating explanations...")
    explanations = generate_example_explanations(val_data, val_preds, val_probs, n_examples=10)
    print(explanations)

    # Save explanations to file
    output_dir = Path(__file__).parent
    with open(output_dir / "EXPLANATIONS.md", 'w') as f:
        f.write("# ED-05 Example Predictions & Explanations\n\n")
        f.write("```\n")
        f.write(explanations)
        f.write("\n```\n")
    print(f"  Explanations saved to {output_dir / 'EXPLANATIONS.md'}")

    # ──────────────────────────────────────────
    # STEP 7: Save CF4 vectorizer for reproducibility
    # ──────────────────────────────────────────
    import pickle
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "cf4_vectorizer.pkl", 'wb') as f:
        pickle.dump(cf4_vectorizer, f)
    print(f"\n  CF4 vectorizer saved to {models_dir / 'cf4_vectorizer.pkl'}")

    # ──────────────────────────────────────────
    # STEP 8: Classification report
    # ──────────────────────────────────────────
    from sklearn.metrics import classification_report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(val_true, val_preds, target_names=CLASS_ORDER))

    print("\n✅ Pipeline complete!")
    print(f"   Estimated total score: {total:.1f}/100")

    return total


if __name__ == '__main__':
    main()
