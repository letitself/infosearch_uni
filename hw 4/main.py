from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymorphy2 import MorphAnalyzer

from preprocess import preprocess_docs, preprocess_corpus, index_matrix
from search import search_query
from score import count_score



def main():
    # task 1
    filename = input('Input your filename: ')
    vec_types = 'bm25, bert'.split(', ')
    vectorization = input('You  want to search with bm25 or bert?')
    if vectorization not in vec_types:
        vectorization = input('bm25, bert?')
    tokenizer = RegexpTokenizer(r'[\w-]+')
    morph = MorphAnalyzer()
    stops = set(stopwords.words('russian'))
    preprocessed_corpus, docs_array, questions, vectorizer = preprocess_corpus(filename, vectorization)

    query = input('Your query: ')
    while query != '':
        for d in search_query(preprocessed_corpus, query, docs_array, vectorization, vectorizer):
            print(d)
        query = input('Your query: ')

    # task 2
    proceed = input('Go to next task? [y/n]')
    if proceed == 'y':
        for vec_type in vec_types:
            vectorization = vec_type
            if vectorization in ['bm25']:
                preprocessed_corpus, docs_array, questions, vectorizer = preprocess_corpus(filename, vectorization)
                preprocessed_questions = index_matrix(questions, vectorizer)
            else:
                preprocessed_corpus, docs_array, questions, vectorizer = preprocess_corpus(filename, vectorization)
                preprocessed_questions = preprocess_docs(questions, vectorization)

            scores = count_score(preprocessed_corpus, preprocessed_questions, vectorization)
            print(f'{vec_type}: {scores}')


if __name__ == "__main__":
    main()