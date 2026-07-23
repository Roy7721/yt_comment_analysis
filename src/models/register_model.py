# register model

import json
import pickle
import mlflow
import logging
import os

# Set up MLflow tracking URI
#mlflow.set_tracking_uri("http://ec2-54-196-109-131.compute-1.amazonaws.com:5000/")


# MLflow / DagShub tracking setup

os.environ["MLFLOW_TRACKING_USERNAME"] = "Roy7721"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bc912b7d58bd2aca05abdc192e010414493a3886"
mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")
#mlflow.set_experiment("LogisticRegression HP Tuning with Custom Features")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and tag it 'staging'."""
    try:
        # model_info['model_path'] is now the artifact path ("sentiment_model"),
        # so this URI resolves to the pyfunc model logged inside the run.
        model_uri = model_info['model_uri']

        # Create a new version under model_name (creates the registered model on first call)
        model_version = mlflow.register_model(model_uri, model_name)

        # MLflow 3: stages are gone — pin an ALIAS to this specific version instead.
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version,
        )

        logger.debug(
            f"Model {model_name} v{model_version.version} registered with alias 'staging'."
        )
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def main():
    try:
        root_dir = get_root_directory()
        model_info_path = os.path.join(root_dir, 'reports', 'model_info.json')
        #model_info_path = 'model_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model#4"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()