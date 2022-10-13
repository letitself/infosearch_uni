import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
from preprocess import index_query, tokenize_lemmatize,cos_similarity

tokenizer = RegexpTokenizer(r'[\w-]+')
morph = MorphAnalyzer()
stops = set(stopwords.words('russian'))


def search_query(preprocessed_corpus, query, docs_array, vectorization='fasttext', vectorizer=None):
    indexed_query = index_query(query, vectorization, vectorizer)
    scores = cos_similarity(preprocessed_corpus, indexed_query, vectorization)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return docs_array[sorted_scores_indx.ravel()][:10]


def index_matrix(matrix, vectorizer=None):
    tl_matrix = [tokenize_lemmatize(doc) for doc in matrix]
    return vectorizer.transform(tl_matrix)