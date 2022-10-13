import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity



def count_score(corpus, questions, vectorization):
    if vectorization == 'bert':
        scores = np.array(torch.matmul(corpus, questions.T))
    else:
        scores = cosine_similarity(corpus, questions)

    count = 0
    for i in range(len(scores)):
        sorted_scores = np.argsort(scores[i], axis=0)[::-1]
        if i in sorted_scores[:5]:
            count += 1

    return count / len(scores)