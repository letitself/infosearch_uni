from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import list_of_files


vectorizer = TfidfVectorizer()


# matrix tf-idf
def index():
    texts, file_names = list_of_files()
    corpus = preprocess(texts)
    X = vectorizer.fit_transform(corpus)
    return X, file_names


# returning vector of user's req, returning vector of req
def vec_request(users_req):
    prep = preprocess([users_req])
    return vectorizer.transform(prep)


# cosinus simularity function
def cosinus(x, vector):
    simularity = cosine_similarity(x, vector)
    return simularity.reshape(-1)
