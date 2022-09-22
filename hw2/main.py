import numpy as np
import nltk
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from request_search import vec_request, cosinus, index

nltk.download("stopwords")
morph = MorphAnalyzer()
stopwords = stopwords.words("russian")
vectorizer = TfidfVectorizer()


# main function return series to your request
def main():
    x, file_names = index()
    while True:
        request = input("Your request(or *stop*)")
        if "stop" not in request:
            names_sorted = []
            vec_rec = vec_request(request)
            list_cos = cosinus(x, vec_rec)
            id_sort = np.argsort(list_cos)[::-1]
            id_sort = id_sort.tolist()
            for i in range(len(file_names)):
                names_sorted.append(file_names[id_sort[i]])
            print("Siutble to your request series in descending order: ")
            print('\n'.join(names_sorted))
        else:
            break


if __name__ == "__main__":
    main()