import json
import pickle
import os
import yaml
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import classification_report,confusion_matrix
import logging
import matplotlib.pyplot as plt 
import seaborn as sns
import mlflow
import joblib
import spacy
import mlflow.pyfunc
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix





os.environ["MLFLOW_TRACKING_USERNAME"] = "Roy7721"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bc912b7d58bd2aca05abdc192e010414493a3886"
mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")
mlflow.set_experiment("dvc-pipeline-runs#1")

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def load_data(root_dir: str):
    """Load data from a CSV file."""
    try:
        X_test = load_npz(os.path.join(root_dir, 'data/processed/X_test.npz'))
        y_test = np.load(os.path.join(root_dir, 'data/processed/y_test.npy'))
        #df = pd.read_csv(file_path)
        #df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', root_dir)
        return X_test , y_test
    except Exception as e:
        logger.error('Error loading data from %s: %s', root_dir, e)
        raise



def load_model(model_dir):

    model = pickle.load(open(model_dir,'rb'))
    logger.debug("model loaded succesfully")
    return model



class SentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # context.artifacts holds the LOCAL paths MLflow copied our files to.
        # The keys ('vectorizer', 'model') are whatever we name them in log_model later.

        # vectorizer.joblib is a DICT: {'tfidf': ..., 'custom_columns': [...]}
        bundle = joblib.load(context.artifacts["vectorizer"])
        self.tfidf = bundle["tfidf"]
        self.custom_columns = bundle["custom_columns"]

        # lor_model.pkl is the Pipeline(MaxAbsScaler -> LogisticRegression)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

        # spaCy loaded ONCE here, not per-comment (it's the slow part)
        self.nlp = spacy.load("en_core_web_sm")
        # Same Store-Python sandbox fix as data_preprocessing.py: keep NLTK data in a
        # plain folder and search it FIRST, else nltk's path-security check rejects the
        # sandbox-redirected AppData path.
        nltk_dir = os.environ.get("NLTK_DATA", r"C:\nltk_data")
        nltk.data.path.insert(0, nltk_dir)
        nltk.download("stopwords", download_dir=nltk_dir, quiet=True)
        nltk.download("wordnet", download_dir=nltk_dir, quiet=True)
        #nltk.download("stopwords", quiet=True)
        #nltk.download("wordnet", quiet=True)
        self.stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        self.lemmatizer = WordNetLemmatizer()

        self.UNIVERSAL_POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                     'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    def _preprocess(self, comment):
        """Identical to data_preprocessing.preprocess_comment — must match training."""
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = ' '.join(w for w in comment.split() if w not in self.stop_words)
        comment = ' '.join(self.lemmatizer.lemmatize(w) for w in comment.split())
        return comment

    def _extract_features(self, text):
        """Identical to build_features.extract_custom_features, using self.nlp."""
        doc = self.nlp(str(text))
        word_list = [t.text for t in doc]
        word_count = len(word_list)
        pos_tags = [t.pos_ for t in doc]
        if word_count > 0:
            pos_proportion = {tag: pos_tags.count(tag) / word_count for tag in self.UNIVERSAL_POS}
        else:
            pos_proportion = {tag: 0 for tag in self.UNIVERSAL_POS}
        return {
            'comment_length': len(str(text)),
            'word_count': word_count,
            'avg_word_length': sum(len(w) for w in word_list) / word_count if word_count > 0 else 0,
            'unique_word_count': len(set(word_list)),
            'lexical_diversity': len(set(word_list)) / word_count if word_count > 0 else 0,
            'pos_count': len(pos_tags),
            **pos_proportion
        }

#4. predict — the actual chain

    def predict(self, context, model_input):
        # The Chrome plugin sends comments as JSON; MLflow serving hands them to us
        # as a DataFrame. Locally you might pass a plain list. Handle both.
        if isinstance(model_input, pd.DataFrame):
            comments = model_input.iloc[:, 0].astype(str).tolist()
        else:
            comments = [str(c) for c in model_input]

        # SAME order as training: preprocess FIRST, then tfidf + custom features on the clean text
        cleaned = [self._preprocess(c) for c in comments]

        tfidf_part = self.tfidf.transform(cleaned)

        feats = pd.DataFrame([self._extract_features(c) for c in cleaned])
        feats = feats.reindex(columns=self.custom_columns, fill_value=0)   # lock column order

        X = hstack([tfidf_part, csr_matrix(feats.values)]).tocsr()

        return self.model.predict(X)

def predict_and_report(x_test,y_test,model_dir):
    try:    
        model = load_model(model_dir)

        y_pred = model.predict(x_test)

        report = classification_report(y_true=y_test,y_pred=y_pred,output_dict=True)

        cm = confusion_matrix(y_test, y_pred)
            
        logger.debug('Model evaluation completed')
    
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()






def save_model_info(run_id: str, model_uri: str, file_path_model_info: str,report,file_path_metrics:str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_uri': model_uri
        }

        metrics = {
                'accuracy':  report['accuracy'],
                'macro_f1':  report['macro avg']['f1-score'],
                'neg_recall': report['-1']['recall'],   # the class you care about most
            }

        for label, metrics in report.items():
                                    if isinstance(metrics, dict):
                                        mlflow.log_metrics({
                                            f"test_{label}_precision": metrics['precision'],
                                            f"test_{label}_recall": metrics['recall'],
                                            f"test_{label}_f1-score": metrics['f1-score']
                                        })
        # Save the dictionary as a JSON file
        with open(file_path_model_info, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path_model_info)
        
        with open(file_path_metrics, 'w') as f:
                json.dump(metrics, f, indent=4)
        logger.debug('Metrics saved to %s', file_path_metrics)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def main():
    with mlflow.start_run() as run:
        try:
            root_dir = get_root_directory()
            MODELS_DIR    = os.path.join(os.path.join(root_dir, 'models') , 'lor_model.pkl')

            #params = load_params(os.path.join(root_dir, 'params.yaml'))

            

            X_test, y_test = load_data(root_dir=root_dir)

            # Log the vectorizer as an artifact
            #mlflow.log_artifact(os.path.join(os.path.join(root_dir, 'models'), 'vectorizer.joblib'))

            report,cm = predict_and_report(x_test=X_test,y_test=y_test,model_dir=MODELS_DIR)

            ARTIFACT_PATH = "sentiment_model"
            logged_model= mlflow.pyfunc.log_model(
                name=ARTIFACT_PATH,
                python_model=SentimentModel(),
                artifacts={
                    "vectorizer": os.path.join(root_dir, "models", "vectorizer.joblib"),
                    "model":      os.path.join(root_dir, "models", "lor_model.pkl"),
                },
            )

            print("model_uri ->", logged_model.model_uri)

            

            file_path_metrics = os.path.join(root_dir, 'reports', 'metrics.json')
            file_path_model_info = os.path.join(root_dir, 'reports', 'model_info.json')

            save_model_info(run.info.run_id, file_path_metrics=file_path_metrics,file_path_model_info=file_path_model_info, report=report,model_uri=logged_model.model_uri,)

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")
            
            print(report)
            # Add important tags
            mlflow.set_tag("model_type", "LogisticRegression")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "Reddit Comments")

        except Exception as e:
                    logger.error(f"Failed to complete model evaluation: {e}")
                    print(f"Error: {e}")
                    raise


if __name__ == '__main__':
    main()