import yaml
import logging
from src.data_loader import prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        train_df, dev_df = prepare_data(config)
        logger.info("Data preparation complete.")
        print(f"Train set: {len(train_df)} samples")
        print(f"Dev set: {len(dev_df)} samples")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")

if __name__ == "__main__":
    main()
