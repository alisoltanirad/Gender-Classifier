# https://github.com/alisoltanirad/Gender-Identifier.git
# Dependencies: nltk
import ssl
import random
import nltk

def main():
    names = get_names()
    classify_gender(names)


def classify_gender(names):
    pass


def get_names():
    download_corpus()
    names = [(name, 'male') for name in nltk.corpus.names.words('male.txt')] + \
            [(name, 'female') for name in nltk.corpus.names.words('female.txt')]
    random.shuffle(names)
    return names


def download_corpus():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('names')


if __name__ == '__main__':
    main()