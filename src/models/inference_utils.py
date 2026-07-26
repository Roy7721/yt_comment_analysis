"""Shared text preprocessing + custom feature extraction."""

import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# The 17 universal POS tags in a FIXED order
UNIVERSAL_POS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

# Sentiment-bearing words we deliberately KEEP (not stripped as stopwords).
_KEEP_WORDS = {"not", "but", "however", "no", "yet"}


def setup_nltk(nltk_dir=None):
    """Point NLTK at a writable folder and ensure the corpora exist.

    Microsoft Store Python sandboxes AppData writes, which trips NLTK's own
    path-security check. Keeping data in a plain folder searched FIRST avoids
    that. Returns the resolved directory.
    """
    nltk_dir = nltk_dir or os.environ.get("NLTK_DATA", r"C:\nltk_data")
    nltk.data.path.insert(0, nltk_dir)
    nltk.download("wordnet", download_dir=nltk_dir, quiet=True)
    nltk.download("stopwords", download_dir=nltk_dir, quiet=True)
    return nltk_dir


def get_stop_words():
    """English stopwords minus the sentiment-bearing words we keep."""
    return set(stopwords.words("english")) - _KEEP_WORDS


def get_lemmatizer():
    """The WordNet lemmatizer used by both training and inference."""
    return WordNetLemmatizer()


def preprocess_comment(comment, stop_words, lemmatizer):
    """Clean one comment. MUST be identical for training and inference."""
    comment = comment.lower().strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
    comment = " ".join(w for w in comment.split() if w not in stop_words)
    comment = " ".join(lemmatizer.lemmatize(w) for w in comment.split())
    return comment


def extract_custom_features(text, nlp):
    """Turn one comment into 6 statistics + 17 POS proportions (fixed key order)."""
    doc = nlp(str(text))
    word_list = [token.text for token in doc]
    word_count = len(word_list)
    pos_tags = [token.pos_ for token in doc]

    if word_count > 0:
        pos_proportion = {
            tag: pos_tags.count(tag) / word_count for tag in UNIVERSAL_POS
        }
    else:
        pos_proportion = {tag: 0 for tag in UNIVERSAL_POS}

    return {
        "comment_length": len(str(text)),
        "word_count": word_count,
        "avg_word_length": sum(len(w) for w in word_list) / word_count
        if word_count > 0
        else 0,
        "unique_word_count": len(set(word_list)),
        "lexical_diversity": len(set(word_list)) / word_count if word_count > 0 else 0,
        "pos_count": len(pos_tags),
        **pos_proportion,
    }
