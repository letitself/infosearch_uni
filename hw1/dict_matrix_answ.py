import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from preprocess import preprocess

# constant names, already in lower register
NAMES = [['моника', 'мон'],
         ['рэйчел', 'рейч', 'рэйч'],
         ['чендлер', 'чэндлер', 'чен'],
         ['фиби', 'фибс'],
         ['росс'],
         ['джоуи', 'джои', 'джо']]

#matrix case solution
def matrix_case(path):  # func wich give us an answers for most freq char, most freq word less freq word and word which contain every document
  corpus = []  # creating empty list for words
  for root, dirs, files in os.walk(path):
      for name in files:
          corpus.append(' '.join(preprocess(os.path.join(root,name))))
  vectorizer = CountVectorizer(analyzer='word')
  X = vectorizer.fit_transform(corpus)

  features = vectorizer.get_feature_names()
  matrix_freq = np.asarray(X.sum(axis=0)).ravel()
  most_freq = features[np.argmax(matrix_freq)]
  less_freq = features[np.argmin(matrix_freq)]

  full = np.apply_along_axis(lambda x: 0 not in x, 0, X.toarray())
  id = np.where(full)[0]
  every_word = [features[i] for i in id]
  #chech wich char is more remarcable
  names_ = {}

  for char in NAMES:
      count = 0
      for name in char:
          index = vectorizer.vocabulary_.get(name.lower())
          if index:
              count += X.T[index].sum()
      names_[char[0]] = 0
      names_[char[0]] += count

  freq_char = sorted(names_.items(), key=lambda x: x[1], reverse=True)[0][0]

  return most_freq, less_freq, every_word, freq_char


def dictionary(texts):
    dictionary = {}
    for keys, values in texts.items():
        for i in values:
            rvr = dictionary.setdefault(i, {})
            rvr[keys] = rvr.get(keys, 0) + 1
    return dictionary


def dict_case(dci_d):
    rotated_dic = dictionary(dci_d)

    counts = {} #dict with freq of word in txt
    for word, documents in rotated_dic.items():
        for document, count in documents.items():
            counts[word] = counts.get(word, 0) + count

    sorted_dict = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    most_freq_d = list(sorted_dict.items())[0][0]
    less_freq_d = list(sorted_dict.items())[-1][0]
    every_word_d = [word for word, documents in rotated_dic.items() if len(documents) == 165]

    #check wich char is more remarcable
    chars = {}
    for c in NAMES:
        count = 0
        for name in c:
            try:
                count += counts[name]
            except KeyError:
                pass
        chars[c[0]] = 0
        chars[c[0]] += count
    freq_char_d = sorted(chars.items(), key=lambda x: x[1], reverse=True)[0][0]

    return most_freq_d, less_freq_d, every_word_d, freq_char_d


