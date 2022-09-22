import os
from pymorphy2 import MorphAnalyzer
from string import punctuation
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download("stopwords")
morph = MorphAnalyzer()
stopwords = stopwords.words("russian")
vectorizer = TfidfVectorizer()


# list of files and texts
def list_of_files():
    file_paths = []
    file_names = []
    texts = []
    data_path = input('Your data path')
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name[0] != '.':
                file_paths.append(os.path.join(root, name))
                file_names.append(name)
    for file_path in file_paths:
        with open(file_path, 'r', errors='ignore') as f:
            text = f.read()
        texts.append(text)

    return texts, file_names


# preprocessing
def preprocess(texts):
    pps_texts = []
    for text in texts:
        pps_texts.append(' '.join(morph.parse(w.strip(punctuation))[0].normal_form for w in text.split()
                                           if w not in stopwords))
    return pps_texts
