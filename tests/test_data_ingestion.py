#test data_ingestion

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    output_path = ingestion.initiate_data_ingestion()
    print(f"Data saved to: {output_path}")