# https://github.com/alisoltanirad/Gender-Identifier.git
# Dependencies: nltk
import ssl
import random
import string
import nltk

def main():
    classify_gender(get_names())


def classify_gender(names):
    train_set, test_set = preprocess_data(names)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    evaluate_classifier(classifier, test_set)


def evaluate_classifier(classifier, test_set):
    classifier.show_most_informative_features(50)
    print('\nEvaluation\n\t- Accuracy: {:.2%}'.format(
        nltk.classify.accuracy(classifier, test_set)))


def preprocess_data(names):
    data_set = [(extract_features(name), gender) for (name, gender) in names]
    return split_corpus(data_set)


def split_corpus(data_set):
    test_size = round(len(data_set) / 4)
    return data_set[test_size:], data_set[:test_size]


def extract_features(name):
    features = {
        'last_letter': name[-1].lower(),
        'first_letter': name[0].lower()
    }
    for letter in string.ascii_lowercase:
        features['count({})'.format(letter)] = name.lower().count(letter)
    return features


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