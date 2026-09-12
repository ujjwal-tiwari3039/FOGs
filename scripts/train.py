import yaml
import pandas as pd
import numpy as np
import joblib
import logging
from src.data_loader import prepare_data
from src.embeddings import EmbeddingModel
from src.features import FeatureExtractor
from src.model import ShortAnswerClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data
    train_df, dev_df = prepare_data(config)

    # Sample training data if it's too large to avoid memory issues in constrained environments
    if len(train_df) > 5000:
        logger.info("Sampling training data to 5000 samples for stability.")
        train_df = train_df.sample(5000, random_state=42)

    # Sample dev data for calibration to avoid memory issues
    if len(dev_df) > 2000:
        logger.info("Sampling dev data to 2000 samples for calibration stability.")
        dev_df = dev_df.sample(2000, random_state=42)

    # Map labels to integers
    label_map = {'correct': 0, 'contradictory': 1, 'incorrect': 2}
    y_train = train_df['label'].map(label_map).values
    y_dev = dev_df['label'].map(label_map).values

    # 2. Embeddings
    embedder = EmbeddingModel(model_name=config['model']['embedding_model'], device=config['model']['device'])

    logger.info("Encoding training texts...")
    train_student_emb = embedder.encode(train_df['student'].tolist())
    train_ref_emb = embedder.encode(train_df['reference'].tolist())

    logger.info("Encoding dev texts...")
    dev_student_emb = embedder.encode(dev_df['student'].tolist())
    dev_ref_emb = embedder.encode(dev_df['reference'].tolist())

    # 3. Features
    extractor = FeatureExtractor()
    X_train = extractor.extract(
        train_student_emb, train_ref_emb,
        train_df['student'].tolist(), train_df['reference'].tolist()
    )
    X_dev = extractor.extract(
        dev_student_emb, dev_ref_emb,
        dev_df['student'].tolist(), dev_df['reference'].tolist()
    )

    # 4. Model
    clf = ShortAnswerClassifier(model_type=config['model']['classifier_type'])
    clf.train(X_train, y_train)

    # 5. Calibration
    logger.info("Calibrating model...")
    clf.calibrate(X_dev, y_dev)

    # 6. Save
    import os
    os.makedirs("results", exist_ok=True)
    clf.save(config['paths']['model_save_path'])
    # Save the embedding model name and label map too
    joblib.dump({
        "embedding_model": config['model']['embedding_model'],
        "label_map": label_map,
        "feature_extractor_params": {"include_lexical": False}
    }, "results/metadata.joblib")

    logger.info(f"Model saved to {config['paths']['model_save_path']}")

if __name__ == "__main__":
    main()
