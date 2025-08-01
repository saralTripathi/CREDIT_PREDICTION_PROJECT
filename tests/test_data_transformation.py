import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    transformer = DataTransformation()
    train_path, test_path, preprocessor_path = transformer.initiate_data_transformation()
    print(f"Transformed train saved to: {train_path}")
    print(f"Transformed test saved to: {test_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")