import yaml
import pandas as pd
import numpy as np
import joblib
import json
import logging
from src.data_loader import prepare_data
from src.embeddings import EmbeddingModel
from src.features import FeatureExtractor
from src.model import ShortAnswerClassifier
from src.counterfactuals import CounterfactualGenerator
from src.robustness import calculate_robustness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data
    _, dev_df = prepare_data(config)
    y_true_labels = dev_df['label'].tolist()
    # Map labels to integers
    label_map = {'correct': 0, 'contradictory': 1, 'incorrect': 2}
    y_true = np.array([label_map[l] for l in y_true_labels])

    # 2. Load Model and Metadata
    clf = ShortAnswerClassifier()
    clf.load(config['paths']['model_save_path'])
    metadata = joblib.load("results/metadata.joblib")

    embedder = EmbeddingModel(model_name=metadata['embedding_model'])
    extractor = FeatureExtractor(**metadata['feature_extractor_params'])

    # 3. Counterfactual Generator
    # Fit CF4 on a subset of training or the dev training set
    # Since we don't have a separate 'dev-training' set, we'll use the training set from data_loader
    train_df, _ = prepare_data(config)
    cf_gen = CounterfactualGenerator(corpus=train_df['student'].tolist())

    # 4. Robustness Testing
    # To avoid excessive compute, we can sample dev_df
    sample_size = min(1000, len(dev_df))
    dev_sample = dev_df.sample(sample_size, random_state=42)
    y_true_sample = np.array([label_map[l] for l in dev_sample['label']])

    # Original predictions
    s_emb = embedder.encode(dev_sample['student'].tolist())
    r_emb = embedder.encode(dev_sample['reference'].tolist())
    X_orig = extractor.extract(
        s_emb, r_emb,
        dev_sample['student'].tolist(), dev_sample['reference'].tolist()
    )
    y_prob_orig = clf.predict_proba(X_orig)

    # CF predictions
    y_prob_cfs = {}
    for cf_type in ['cf1', 'cf2', 'cf3', 'cf4']:
        logger.info(f"Generating and predicting {cf_type}...")
        cf_texts = [cf_gen.generate(text)[cf_type] for text in dev_sample['student']]

        # Encode and extract features for CF
        cf_s_emb = embedder.encode(cf_texts)
        X_cf = extractor.extract(
            cf_s_emb, r_emb,
            cf_texts, dev_sample['reference'].tolist()
        )
        y_prob_cfs[cf_type] = clf.predict_proba(X_cf)

    # 5. Metrics
    robust_metrics = calculate_robustness(y_true_sample, y_prob_orig, y_prob_cfs)

    # 6. Save
    with open(config['paths']['robustness_save_path'], 'w') as f:
        json.dump(robust_metrics, f, indent=4)

    logger.info("Robustness testing complete. Report saved.")
    print(json.dumps(robust_metrics, indent=4))

if __name__ == "__main__":
    main()
