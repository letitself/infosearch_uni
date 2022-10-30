import streamlit as st
from pymorphy2 import MorphAnalyzer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
import ssl
from tqdm import tqdm
from nltk.corpus import stopwords
import json
from scipy import sparse
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import time


try:
    _create_unverified_https_context = ssl._create_unverified_context
except (AttributeError, TypeError):
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
morph = MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

@st.cache(allow_output_mutation=True)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    return tokenizer, model


@st.cache(allow_output_mutation=True)
def parse_files():
    filepath = ('/Users/tarasandrushko/Desktop/project/questions_about_love.jsonl')
    with open(filepath, 'r') as f:
        corpus = list(f)[:10000]

    answers = []
    for i in tqdm(range(10000)):
        question = json.loads(corpus[i])
        answers_to_q = question['answers']
        if answers_to_q != []:
            answers_values = []
            for ans in answers_to_q:
                answer = ans['text']
                if ans['author_rating']['value'] != '':
                    value = int(ans['author_rating']['value'])
                else:
                    value = 0
                answers_values.append((answer, value))

            answers_values.sort(key=lambda x: x[1])
            answers.append(answers_values[-1][0])

    return answers


def preprocessing(text):
    prep_text = []
    for word in text.split():
        word = morph.parse(word.strip(punctuation))[0].normal_form
        if word not in russian_stopwords:
            prep_text.append(word)

    return ' '.join(prep_text)


@st.cache(allow_output_mutation=True)
def countvectorizer_index(prep_answers):
    count_vectorizer = CountVectorizer(analyzer='word')
    return count_vectorizer, normalize(count_vectorizer.fit_transform(prep_answers))


def vec_request_countvectorizer(prep_req, count_vectorizer):
    return normalize(count_vectorizer.transform([prep_req]))


@st.cache(allow_output_mutation=True)
def tfidf_vectorizer_index(prep_answers):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    return tfidf_vectorizer, tfidf_vectorizer.fit_transform(prep_answers)


def vec_request_tfidf_vectorizer(prep_req, tfidf_vectorizer):
    return tfidf_vectorizer.transform([prep_req])


@st.cache(allow_output_mutation=True)
def bm_25(corpus):
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer_2 = TfidfVectorizer(use_idf=True, norm='l2')
    count_vectorizer_2 = CountVectorizer(analyzer='word')
    x_count_vec = count_vectorizer_2.fit_transform(corpus)
    x_tf_vec = tf_vectorizer.fit_transform(corpus)
    x_idf_vec = tfidf_vectorizer_2.fit_transform(corpus)
    idf = tfidf_vectorizer_2.idf_
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl))
    rows = []
    cols = []
    bm25 = []
    for i, j in tqdm(zip(*x_tf_vec.nonzero())):
        rows.append(i)
        cols.append(j)
        A = idf[j] * x_tf_vec[i, j] * (k + 1)
        B = x_tf_vec[i, j] + B_1[i]
        bm25.append(float(A / B))

    matrix = sparse.csr_matrix((bm25, (rows, cols)))
    return count_vectorizer_2, matrix


def vec_request_bm25(prep_req, count_vectorizer_2):
    return count_vectorizer_2.transform([prep_req])


@st.cache(allow_output_mutation=True)
def load_fasttext():
    return np.load('fasttext_matrix.npy')


def vec_request_fasttext(request, fasttext_model):
    lemmas = request.split()
    request_vec = np.zeros((len(lemmas), 300))
    for i in range(len(lemmas)):
        if (lemmas[i] in fasttext_model) and (lemmas[i] != ''):
            request_vec[i] = fasttext_model[lemmas[i]]

    request_vec = np.mean(request_vec, axis=0)
    request_vec = np.nan_to_num(request_vec)
    request_vec = request_vec.reshape(1, -1)
    return normalize(request_vec)


@st.cache(allow_output_mutation=True)
def rubert_index():
    return np.load('rubert_matrix.npy')


def vec_request_bert(request, model, tokenizer):
    t = tokenizer(request, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu()


def similarity(request, matrix):
    return matrix.dot(request.T)


@st.cache(allow_output_mutation=True)
def preproc_answers(answers):
    return [preprocessing(a) for a in answers]

def main():
    st.title('final project, questions about love')
    answers = parse_files()
    prep_answers = preproc_answers(answers)
    tok_bert, bert_model = load_models()
    count_vectorizer, countvectorizer_matrix = countvectorizer_index(prep_answers)
    tfidf_vectorizer, tfidf_vectorizer_matrix = tfidf_vectorizer_index(prep_answers)
    count_vectorizer_2, bm_25_matrix = bm_25(prep_answers)
    fasttext_matrix = load_fasttext()
    bert_matrix = rubert_index()
    bert_matrix = sparse.csr_matrix(bert_matrix)

    algorithm = st.radio("Choose model", ('TfidfVectorizer', 'bm25', 'bert'))
    st.subheader('Your query:')
    request = st.text_input('')
    prep_req = preprocessing(request)
    button = st.button('Search', key='1')
    time_start = time.time()
    if button == 1:

        if algorithm == 'TfidfVectorizer':
            vec_rec_tf = vec_request_tfidf_vectorizer(prep_req, tfidf_vectorizer)
            sim_req_tf = similarity(vec_rec_tf, tfidf_vectorizer_matrix)
            id_sort = np.argsort(sim_req_tf.toarray(), axis=0)[::-1]

        elif algorithm == 'bm25':
            vec_rec_bm = vec_request_bm25(prep_req, count_vectorizer_2)
            sim_req_bm = similarity(vec_rec_bm, bm_25_matrix)
            id_sort = np.argsort(sim_req_bm.toarray(), axis=0)[::-1]

        else:
            vec_rec_bert = vec_request_bert(request, bert_model, tok_bert)
            sim_req_bert = similarity(vec_rec_bert, bert_matrix)
            id_sort = np.argsort(sim_req_bert, axis=0)[::-1]

        st.write("5 most common: ")
        st.write('\n\n'.join(np.array(answers)[id_sort.ravel()][:5]))

        time_end = time.time()
        time_count = time_end - time_start
        st.write("Search time: ", time_count)


if __name__ == "__main__":
    main()
