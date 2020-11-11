# https://github.com/alisoltanirad/Gender-Identifier.git
# Dependencies: nltk
import random
import nltk

def main():
    names = get_names()
    classify_gender(names)


def classify_gender(names):
    pass


def get_names():
    nltk.download('names')
    names = [(name, 'male') for name in nltk.corpus.names.words('male.txt')] + \
            [(name, 'female') for name in nltk.corpus.names.words('female.txt')]
    random.shuffle(names)
    return names


if __name__ == '__main__':
    main()