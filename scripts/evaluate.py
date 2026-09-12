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
from src.evaluation import evaluate_model
from src.calibration import get_calibration_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data
    _, dev_df = prepare_data(config)

    # Sample dev data to avoid memory issues
    if len(dev_df) > 5000:
        logger.info("Sampling dev data to 5000 samples for evaluation stability.")
        dev_df = dev_df.sample(5000, random_state=42)

    label_map = {'correct': 0, 'contradictory': 1, 'incorrect': 2}
    y_true = dev_df['label'].map(label_map).values

    # 2. Load Model and Metadata
    clf = ShortAnswerClassifier()
    clf.load(config['paths']['model_save_path'])
    metadata = joblib.load("results/metadata.joblib")

    embedder = EmbeddingModel(model_name=metadata['embedding_model'])
    extractor = FeatureExtractor(**metadata['feature_extractor_params'])

    # 3. Feature Extraction
    logger.info("Encoding dev texts...")
    dev_student_emb = embedder.encode(dev_df['student'].tolist())
    dev_ref_emb = embedder.encode(dev_df['reference'].tolist())

    X_dev = extractor.extract(
        dev_student_emb, dev_ref_emb,
        dev_df['student'].tolist(), dev_df['reference'].tolist()
    )

    # 4. Predictions
    y_prob = clf.predict_proba(X_dev)
    y_pred = np.argmax(y_prob, axis=1)

    # 5. Metrics
    perf_metrics = evaluate_model(y_true, y_pred, y_prob)
    calib_metrics = get_calibration_metrics(y_true, y_prob)

    final_metrics = {**perf_metrics, **calib_metrics}

    # 6. Save
    with open(config['paths']['metrics_save_path'], 'w') as f:
        json.dump(final_metrics, f, indent=4)

    logger.info("Evaluation complete. Metrics saved.")
    print(json.dumps(final_metrics, indent=4))

if __name__ == "__main__":
    main()
