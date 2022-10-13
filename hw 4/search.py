import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import index_query, tokenize_lemmatize

tokenizer = RegexpTokenizer(r'[\w-]+')
morph = MorphAnalyzer()
stops = set(stopwords.words('russian'))

def mean_pooling(model_output):
    return model_output[0][:, 0]


def bert_vec(docs):
    tokenizer = AutoTokenizer.from_pretrained("sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sbert_large_nlu_ru")

    encoded_input = tokenizer(docs, padding=True, truncation=True, max_length=24, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output)

def cos_similarity(source_matrix, indexed_query, vectorization='fasttext'):
    if vectorization == 'bert':
        exp = np.expand_dims(indexed_query[0], axis=0)
        cm_matrix = cosine_similarity(source_matrix.numpy(), exp)
    if vectorization != 'bm25':
        cm_matrix = cosine_similarity(source_matrix, indexed_query)
    else:
        cm_matrix = np.dot(source_matrix, indexed_query.T).toarray()

    return cm_matrix


def search_query(preprocessed_corpus, query, docs_array, vectorization='fasttext', vectorizer=None):
    indexed_query = index_query(query, vectorization, vectorizer)
    scores = cos_similarity(preprocessed_corpus, indexed_query, vectorization)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return docs_array[sorted_scores_indx.ravel()][:10]


def index_matrix(matrix, vectorizer=None):
    tl_matrix = [tokenize_lemmatize(doc) for doc in matrix]
    return vectorizer.transform(tl_matrix)