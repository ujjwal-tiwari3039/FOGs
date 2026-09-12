import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_xml_file(file_path):
    """
    Parses a single XML file and extracts student answers and reference answers.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Get the first BEST reference answer
        ref_ans = None
        for ra in root.findall(".//referenceAnswer"):
            if ra.get("category") == "BEST":
                ref_ans = ra.text
                break

        # If no BEST, take the first available reference answer
        if ref_ans is None:
            ra_first = root.find(".//referenceAnswer")
            if ra_first is not None:
                ref_ans = ra_first.text

        if ref_ans is None:
            return []

        results = []
        for sa in root.findall(".//studentAnswer"):
            accuracy = sa.get("accuracy")
            text = sa.text
            if accuracy and text:
                # Ensure labels are exactly as requested
                label = accuracy.strip().lower()
                if label in ["correct", "incorrect", "contradictory"]:
                    results.append({
                        "reference": ref_ans,
                        "student": text,
                        "label": label
                    })
        return results
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return []

def load_dataset(root_dir):
    """
    Walks through the directory and parses all XML files.
    """
    all_data = []
    xml_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))

    logger.info(f"Found {len(xml_files)} XML files in {root_dir}")

    for file_path in tqdm(xml_files, desc=f"Parsing {os.path.basename(root_dir)}"):
        all_data.extend(parse_xml_file(file_path))

    return pd.DataFrame(all_data)

def prepare_data(config):
    """
    Main function to load train and dev sets and save them as CSVs.
    """
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']

    # SemEval 2013 Task 7 structure: training/ and test/
    train_path = os.path.join(raw_dir, "semeval_3way/training")
    dev_path = os.path.join(raw_dir, "semeval_3way/test")

    logger.info("Loading training data...")
    train_df = load_dataset(train_path)

    logger.info("Loading development data...")
    dev_df = load_dataset(dev_path)

    # Save to CSV for consistency
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(processed_dir, "dev.csv"), index=False)

    logger.info(f"Processed data saved. Train size: {len(train_df)}, Dev size: {len(dev_df)}")
    return train_df, dev_df
