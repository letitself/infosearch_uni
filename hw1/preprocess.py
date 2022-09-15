import nltk
from nltk.corpus import stopwords
import pymorphy2
import re

nltk.download('stopwords')
stopwords = stopwords.words("russian")#stop words
morph = pymorphy2.MorphAnalyzer() #lemmatizer

TOKEN_RE = re.compile(r'[а-яё]+')


def preprocess(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        text = ''.join(f.readlines()[:-4]) #removing last 5 strings with subtitle meta information
    text = text.lower() #lower case
    text = re.sub(r'[^\w\s]', '', text) #punktuation
    drt_tokens = text.split() # creating a list of tokens
    tokens = [] # empty list for proceed tokens
    for i in drt_tokens:
      if i.isalpha() and i not in stopwords:
        i = morph.normal_forms(i.strip())[0]
        tokens.append(i)


    return tokens
