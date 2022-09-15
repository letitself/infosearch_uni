
from dict_matrix_answ import matrix_case, dict_case
from preprocess import preprocess
import os


def main(path):

    #creating a dict for dict_case

    dci_d = {}
    i = 1
    for root, dirs, files in os.walk(path):
        for name in files:
            preprocessed = preprocess(os.path.join(root, name))
            dci_d[i] = preprocessed
            i += 1

    most_freq, less_freq, every_word, freq_char = matrix_case(path)
    most_freq_d, less_freq_d, every_word_d, freq_char_d = dict_case(dci_d)

    print(most_freq, ' - самое частотное слово для матрицы')
    print(less_freq, ' - самое не частотное слово для матрицы')
    print(every_word, ' - слово встречающееся в каждом файле')
    print(freq_char, ' - самый частотный персонаж')
    print(most_freq_d, ' - самое частотное слово для матрицы')
    print(less_freq_d, ' - самое не частотное слово для матрицы')
    print(every_word_d, ' - слово встречающееся в каждом файле')
    print(freq_char_d, ' - самый частотный персонаж')

    return


path = input('Введите путь к папке')
finish = main(path)
print(finish)
