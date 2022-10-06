import nltk
import json
import numpy as np
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from scipy import sparse


morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')


def preprocess(txt):
  txt = tokenizer.tokenize(txt.lower())
  lemma = [morph.parse(i)[0].normal_form for i in txt
           if i not in stop_words and i not in punctuation]
  return ' '.join(lemma)


def corpus_create(path):
    with open(path, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    lemmas = []
    txts = []
    for i in corpus:
        answers = json.loads(i)['answers']
        if answers:
            values = np.array(map(int, [i['author_rating']['value'] for i in answers if i != '']))
            answer = answers[np.argmax(values)]['text']
            lemmas.append(preprocess(answer))
            txts.append(answer)
    return lemmas, txts


def indexes(corpus, k=2, b=0.75):
    x_count = count_vectorizer.fit_transform(corpus)
    x_idf = tfidf_vectorizer.fit_transform(corpus)
    x_tf = tf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_
    len_d = x_count.sum(axis=1)
    avdl = len_d.mean()
    fin = k * (1 - b + b * len_d / avdl)
    matrix = sparse.lil_matrix(x_tf.shape)
    for i, j in zip(*x_tf.nonzero()):
        matrix[i, j] = (x_tf[i, j] * (k + 1) * idf[j])/(x_tf[i, j] + fin[i])
    return matrix.tocsr()


def qr_indexes(qr):
    #vectorizing user's query
    return count_vectorizer.transform([qr])


def bm25_search(qr, corpus):
    #similarity counting
    return corpus.dot(qr.T)


def find(qr, corpus, answers):
    lemmas = preprocess(qr)
    if lemmas:
        qr_index = qr_indexes(lemmas)
        bm25 = bm25_search(qr_index, corpus)
        ind = np.argsort(bm25.toarray(), axis=0)
        return np.array(answers)[ind][::-1].squeeze()
    else:
        pass