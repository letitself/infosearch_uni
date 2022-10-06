from preprocess import indexes, corpus_create, find



def main():
    corpus, lemmas = corpus_create(input('Your file: '))
    matrix = indexes(lemmas)
    qr = input('Your query: ')
    while qr != '':
        result = find(qr, matrix, corpus)
        print(*result[:20])
        print('Anouther one?')
        qr = input('Your query: ')


if __name__ == '__main__':
    main()