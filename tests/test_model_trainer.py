import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.components.model_trainer import ModelTrainer
import numpy as np

if __name__ == "__main__":
    # Load processed data as dicts with 'X' and 'y'
    train_data = np.load("artifacts/train.npy", allow_pickle=True).item()
    test_data = np.load("artifacts/test.npy", allow_pickle=True).item()
    x_train, y_train = train_data['X'], train_data['y']
    x_test, y_test = test_data['X'], test_data['y']
    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]

    trainer = ModelTrainer()
    model_path = trainer.initiate_model_trainer(x_train, y_train, x_test, y_test)
    print(f"Model saved at: {model_path}")