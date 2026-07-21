# src/data/data_preprocessing.py

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Build these ONCE at import time rather than rebuilding them for every comment.
# Previously both were constructed inside preprocess_comment, i.e. ~36k times.
STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
LEMMATIZER = WordNetLemmatizer()

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        comment = ' '.join([word for word in comment.split() if word not in STOP_WORDS])

        # Lemmatize the words
        comment = ' '.join([LEMMATIZER.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def drop_empty_comments(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Drop rows whose comment became empty during preprocessing.

    Stripping symbols and stopwords can reduce a comment to an empty string. An empty
    string is NOT null in memory, so isnull() reports zero here - but to_csv writes it
    as a blank field and read_csv reads that blank back as NaN. That round-trip is where
    surprise null values downstream come from, so drop these rows before saving.
    """
    try:
        before = len(df)
        df = df[df['clean_comment'].notna()]
        df = df[df['clean_comment'].str.strip() != '']
        removed = before - len(df)
        logger.debug('%s: dropped %d empty comments after preprocessing (%d -> %d rows)',
                     split_name, removed, before, len(df))
        return df
    except Exception as e:
        logger.error(f"Error while dropping empty comments: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Remove rows that became empty during preprocessing, so they don't reappear
        # as NaN when these CSVs are read back downstream.
        train_processed_data = drop_empty_comments(train_processed_data, 'train')
        test_processed_data = drop_empty_comments(test_processed_data, 'test')

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")
        raise  # re-raise so the process exits non-zero and DVC marks the stage FAILED

if __name__ == '__main__':
    main()