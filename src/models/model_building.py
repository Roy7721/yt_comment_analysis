# src/model/model_building.py

import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
import scipy
from scipy.sparse import load_npz

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise






def train_lor(X_train, y_train, C: float, max_iter: int, l1_ratio: int) -> LogisticRegression:
    """Train a LogisticRegression model."""
    try:
        best_model = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('clf', LogisticRegression(C=C,
            l1_ratio=l1_ratio,
            class_weight=None,
            solver='saga',
            max_iter=max_iter,
            random_state=42))
            ])
        best_model.fit(X_train, y_train)
        logger.debug('LoR model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during LoR model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        #max_features = params['model_building']['max_features']
        #ngram_range = tuple(params['model_building']['ngram_range'])

        C = params['model_building']['C']
        l1_ratio = params['model_building']['l1_ratio']
        max_iter = params['model_building']['max_iter']

        # Load the preprocessed training data from the interim directory
        X_train = load_npz(os.path.join(root_dir, 'data/processed/X_train.npz'))
        #X_test = load_npz(os.path.join(root_dir, 'data/processed/X_test.npz'))
        y_train = np.load(os.path.join(root_dir, 'data/processed/y_train.npy'))
        #y_test = np.load(os.path.join(root_dir, 'data/processed/y_train.npz'))

        # Apply TF-IDF feature engineering on training data
        #X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_lor(X_train, y_train, C=C, max_iter=max_iter, l1_ratio=l1_ratio)

        MODELS_DIR    = os.path.join(root_dir, 'models')

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(MODELS_DIR , 'lor_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()