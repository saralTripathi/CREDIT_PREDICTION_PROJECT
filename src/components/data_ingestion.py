import os
import pandas as pd
from src.constant import APPLICATION_TRAIN_PATH
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join("artifacts")
    train_file_name: str = "application_train.csv"

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        os.makedirs(self.config.artifact_folder, exist_ok=True)
        dst_path = os.path.join(self.config.artifact_folder, self.config.train_file_name)
        df = pd.read_csv(APPLICATION_TRAIN_PATH)
        df.to_csv(dst_path, index=False)
        logging.info(f"Train data ingested and saved to: {dst_path}")
        return dst_path