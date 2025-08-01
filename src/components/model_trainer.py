import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join("artifacts")
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            'XGBClassifier': XGBClassifier(n_jobs=-1, verbosity=0),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=-1)
        }
        self.model_param_grid = self.utils.read_yaml_file(self.config.model_config_file_path)["model_selection"]["model"]

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        report = {}
        for name, model in self.models.items():
            print(f"Training base model: {name} ...")
            logging.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            report[name] = score
            logging.info(f"{name} accuracy: {score:.4f}")
            print(f"{name} accuracy: {score:.4f}")
        logging.info(f"Model evaluation report: {report}")
        print(f"Model evaluation report: {report}")
        return report

    def finetune_best_model(self, model_name, model, X_train, y_train):
        print(f"Starting GridSearchCV for {model_name} ...")
        logging.info(f"Starting GridSearchCV for {model_name} ...")
        param_grid = self.model_param_grid[model_name]["search_param_grid"]
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best params for {model_name}: {best_params}")
        logging.info(f"Best params for {model_name}: {best_params}")
        model.set_params(**best_params)
        return model

    def initiate_model_trainer(self, x_train, y_train, x_test, y_test):
        try:
            logging.info("Evaluating base models")
            model_report = self.evaluate_models(x_train, y_train, x_test, y_test)
            best_model_name = max(model_report, key=model_report.get)
            best_model = self.models[best_model_name]

            logging.info(f"Best base model: {best_model_name} with accuracy: {model_report[best_model_name]:.4f}")

            # Fine-tune best model
            best_model = self.finetune_best_model(best_model_name, best_model, x_train, y_train)
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            final_score = accuracy_score(y_test, y_pred)
            logging.info(f"Final {best_model_name} test accuracy after tuning: {final_score:.4f}")

            if final_score < self.config.expected_accuracy:
                raise Exception(f"No model found with accuracy above threshold {self.config.expected_accuracy}")

            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            self.utils.save_object(self.config.trained_model_path, best_model)
            logging.info(f"Best model saved at: {self.config.trained_model_path}")

            return self.config.trained_model_path

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e