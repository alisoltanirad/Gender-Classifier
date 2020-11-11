# https://github.com/alisoltanirad/Gender-Identifier.git
# Dependencies: nltk
import ssl
import random
import nltk

def main():
    names = get_names()
    classify_gender(names)


def classify_gender(names):
    data_set = [(extract_features(name), gender) for (name, gender) in names]
    train_set, validation_set, test_set = split_corpus(data_set)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # classifier.train(train_set)

    analyze_classifier(classifier, validation_set)
    evaluate_classifier(classifier, test_set)


def analyze_classifier(classifier, validation_set):
    #classifier.show_most_informative_features(classifier)
    show_errors(classifier, validation_set)


def show_errors(classifier, validation_set):
    errors = get_errors(classifier, validation_set)
    print('- Errors:')
    for (gender, guess, features) in errors:
        print(type(gender))
        print('correct:{:<8} guess:{:<8} features:{}'.format(gender, guess,
                                                              features))


def get_errors(classifier, validation_set):
    errors = []
    for (features, gender) in validation_set:
        guess = classifier.classify(features)
        if guess != gender:
            errors.append((gender, guess, features))

    return errors


def evaluate_classifier(classifier, test_set):
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print('- Evaluation:\nAccuracy: {:.2%}'.format(accuracy))


def extract_features(name):
    features = {
        'last_letter': name[-1].lower()
    }
    return features


def split_corpus(data_set):
    test_set = data_set[-round(len(data_set)/5):]
    train_validation = data_set[:-round(len(data_set)/5)]
    train_set = train_validation[:-round(len(train_validation)/5)]
    validation_set = train_validation[-round(len(train_validation)/5):]
    return train_set, validation_set, test_set


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