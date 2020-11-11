# https://github.com/alisoltanirad/Gender-Identifier.git
# Dependencies: nltk
import ssl
import random
import string
import nltk

def main():
    names = get_names()
    classify_gender(names)


def classify_gender(names):
    data_set = [(extract_features(name), gender) for (name, gender) in names]
    train_set, test_set = split_corpus(data_set)

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    analyze_classifier(classifier)
    evaluate_classifier(classifier, test_set)


def analyze_classifier(classifier):
    classifier.show_most_informative_features(50)


def evaluate_classifier(classifier, test_set):
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print('\nEvaluation\n\t- Accuracy: {:.2%}'.format(accuracy))


def extract_features(name):
    features = {
        'last_letter': name[-1].lower(),
        'first_letter': name[0].lower()
    }
    for letter in string.ascii_lowercase:
        features['count({})'.format(letter)] = name.lower().count(letter)
    return features


def split_corpus(data_set):
    train_set = data_set[:-round(len(data_set)/4)]
    test_set = data_set[-round(len(data_set)/4):]
    return train_set, test_set


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