# https://github.com/alisoltanirad/Gender-Classifier
# Dependencies: nltk
import ssl
import random
import string
import nltk

class GenderClassifier:

    def __init__(self):
        self._prepare_dataset()
        self._train_classifier()

    def classify(self, name):
        return self.classifier.classify(self._extract_features(name))

    def evaluate(self):
        accuracy = nltk.classify.accuracy(self.classifier, self._test_set)
        most_informative_features = self.classifier.most_informative_features(50)
        evaluation_data = {
            'Accuracy': accuracy,
            'Most_Informative_Features': most_informative_features,
        }
        return evaluation_data

    def _train_classifier(self):
        self.classifier = nltk.NaiveBayesClassifier.train(self._train_set)

    def _prepare_dataset(self):
        self._download_corpus()
        self._data_set = [(self._extract_features(name), gender)
                          for (name, gender) in self._get_names()]
        test_size = round(len(self._data_set) / 4)
        self._train_set = self._data_set[test_size:]
        self._test_set = self._data_set[:test_size]

    def _extract_features(self, name):
        features = {
            'last_letter': name[-1].lower(),
            'first_letter': name[0].lower()
        }
        for letter in string.ascii_lowercase:
            features['count({})'.format(letter)] = name.lower().count(letter)
        return features

    def _get_names(self):
        names = [(name, 'male') for name in
                 nltk.corpus.names.words('male.txt')] + \
                [(name, 'female') for name in
                 nltk.corpus.names.words('female.txt')]
        random.shuffle(names)
        return names

    def _download_corpus(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download('names')


def main():
    classifier = GenderClassifier()
    print(classifier.evaluate()['Accuracy'])


if __name__ == '__main__':
    main()