
import spacy
import joblib
import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz

# logging configuration
logger = logging.getLogger('build_features')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('build_features_errors.log')
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


def load_data(interim_dir: str):
    """Load the preprocessed train/test splits from the data_preprocessing stage."""
    try:
        train = pd.read_csv(os.path.join(interim_dir, 'train_processed.csv'))
        test  = pd.read_csv(os.path.join(interim_dir, 'test_processed.csv'))

        # Guard: a comment reduced to empty during preprocessing has no signal,
        # and would also crash spaCy. Drop any that slipped through.
        for name, df in [('train', train), ('test', test)]:
            before = len(df)
            df.dropna(subset=['clean_comment'], inplace=True)
            df.drop(df[df['clean_comment'].astype(str).str.strip() == ''].index, inplace=True)
            if before != len(df):
                logger.debug('%s: dropped %d empty rows', name, before - len(df))

        logger.debug('Loaded train %s, test %s', train.shape, test.shape)
        return train, test
    except Exception as e:
        logger.error('Failed to load interim data: %s', e)
        raise



# All 17 universal POS tags, in a FIXED order -> every comment yields the
# same 17 columns in the same order (train, test, and future inference).
UNIVERSAL_POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']


def extract_custom_features(text, nlp):
    """Turn one comment into 6 statistics + 17 POS proportions."""
    doc = nlp(str(text))
    word_list = [token.text for token in doc]

    comment_length    = len(str(text))
    word_count        = len(word_list)
    avg_word_length   = sum(len(w) for w in word_list) / word_count if word_count > 0 else 0
    unique_word_count = len(set(word_list))
    lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
    pos_count         = len([token.pos_ for token in doc])

    pos_tags = [token.pos_ for token in doc]
    if word_count > 0:
        pos_proportion = {tag: pos_tags.count(tag) / word_count for tag in UNIVERSAL_POS}
    else:
        pos_proportion = {tag: 0 for tag in UNIVERSAL_POS}

    return {
        'comment_length': comment_length,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'unique_word_count': unique_word_count,
        'lexical_diversity': lexical_diversity,
        'pos_count': pos_count,
        **pos_proportion
    }

def build_features(train, test, ngram_range, max_features, use_spacy):
    """TF-IDF (+ optional spaCy custom features) -> sparse train/test matrices."""
    try:
        X_train_text = train['clean_comment'].astype(str)
        X_test_text  = test['clean_comment'].astype(str)

        # --- TF-IDF: fit on TRAIN ONLY, then transform both ---
        tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        X_train = tfidf.fit_transform(X_train_text)
        X_test  = tfidf.transform(X_test_text)
        logger.debug('TF-IDF done. vocab=%d, train=%s', len(tfidf.vocabulary_), X_train.shape)

        custom_columns = None

        if use_spacy:
            logger.debug('Extracting spaCy custom features (this is the slow step)...')
            nlp = spacy.load('en_core_web_sm')

            train_feats = pd.DataFrame([extract_custom_features(t, nlp) for t in X_train_text])
            test_feats  = pd.DataFrame([extract_custom_features(t, nlp) for t in X_test_text])

            # Lock column order; align test to train (defensive - order is already fixed)
            custom_columns = list(train_feats.columns)
            test_feats = test_feats.reindex(columns=custom_columns, fill_value=0)

            # Combine WITHOUT densifying - TF-IDF stays sparse
            X_train = hstack([X_train, csr_matrix(train_feats.values)]).tocsr()
            X_test  = hstack([X_test,  csr_matrix(test_feats.values)]).tocsr()
            logger.debug('Combined with custom features. train=%s', X_train.shape)

        y_train = train['category'].values
        y_test  = test['category'].values

        return X_train, X_test, y_train, y_test, tfidf, custom_columns
    except Exception as e:
        logger.error('Failed to build features: %s', e)
        raise

def save_features(X_train, X_test, y_train, y_test, out_dir):
    """Persist the feature matrices and labels (regenerated each run)."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        save_npz(os.path.join(out_dir, 'X_train.npz'), X_train)
        save_npz(os.path.join(out_dir, 'X_test.npz'),  X_test)
        np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(out_dir, 'y_test.npy'),  y_test)
        logger.debug('Saved feature matrices to %s', out_dir)
    except Exception as e:
        logger.error('Failed to save features: %s', e)
        raise


def save_vectorizer(tfidf, custom_columns, models_dir):
    """Persist the fitted vectorizer + column order (reused at inference)."""
    try:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump({'tfidf': tfidf, 'custom_columns': custom_columns},
                    os.path.join(models_dir, 'vectorizer.joblib'))
        logger.debug('Saved vectorizer to %s', models_dir)
    except Exception as e:
        logger.error('Failed to save vectorizer: %s', e)
        raise

def main():
    # Paths anchored to THIS FILE, not to wherever you run python from.
    ROOT          = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
    INTERIM_DIR   = os.path.join(ROOT, 'data', 'interim')
    PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
    MODELS_DIR    = os.path.join(ROOT, 'models')

    params       = load_params(os.path.join(ROOT, 'params.yaml'))['feature_engineering']
    ngram_range  = tuple(params['ngram_range'])     # [1,2] -> (1,2)
    max_features = params['max_features']
    use_spacy    = params['use_spacy']

    

    train, test = load_data(INTERIM_DIR)

    X_train, X_test, y_train, y_test, tfidf, custom_columns = build_features(
                train, test, ngram_range, max_features, use_spacy
            )

    save_features(X_train, X_test, y_train, y_test, PROCESSED_DIR)
    save_vectorizer(tfidf, custom_columns, MODELS_DIR)

    logger.debug('Feature engineering complete.')



if __name__ == '__main__':
    main()