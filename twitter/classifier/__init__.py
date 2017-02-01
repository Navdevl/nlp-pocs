# -*- coding: utf-8 -*-
import yaml
import nltk
from sklearn.externals import joblib
from nltk.collocations import *
from nltk.metrics import *

from paths import POSPATH, NEGPATH, N_FEATURES, apipath

def load_yaml_files():
    posyml = []
    with open(POSPATH, 'r') as posfile:
        posyml = yaml.load(posfile)

    negyml = []
    with open(NEGPATH, 'r') as negfile:
        negyml = yaml.load(negfile)

    return posyml, negyml

POSYML, NEGYML = load_yaml_files()
CLASSIFIER_FILE = apipath +'/classifier'

class NBClassifier():
    def __init__(self):
        self.trigram_measures = nltk.collocations.TrigramAssocMeasures()
        self.classifier = ''
        self.get_all_words()
        self.top_words()
        self.top_trigrams()
        self.load()
        if self.classifier is '' :
            self.train()
        self.save()

    def get_all_words(self):
        self.all_words = []
        for pos in POSYML:
            self.all_words += [word for word in nltk.word_tokenize(pos['text'])]

        for neg in NEGYML:
            self.all_words += [word for word in nltk.word_tokenize(neg['text'])]

    def top_trigrams(self, n=N_FEATURES):
        finder = TrigramCollocationFinder.from_words(self.all_words)
        finder.apply_freq_filter(3)

        # ignoring all trigrams which occur less than n times
        # freq_trigrams = finder.nbest(self.trigram_measures.pmi, n)
        freq_trigrams = finder.nbest(self.trigram_measures.likelihood_ratio, n)
        # freq_trigrams = finder.above_score(self.trigram_measures.likelihood_ratio,70)
        self.trigram_features = dict([(trigram, True) for trigram in freq_trigrams])

    def top_words(self, n=N_FEATURES):
        freq_words = nltk.FreqDist(word for word in self.all_words)
        # self.word_features = list(self.all_words)[:n]
        freq_words = list(self.all_words)[:n]

        self.word_features = dict([(word, True) for word in freq_words])

    def document_features(self, sentence):
        # features = {}
        # words = nltk.word_tokenize(sentence)
        # for word in self.word_features:
        #     features[word] = (word in words)
        #
        # return features
        features = {}

        sentence_words = nltk.word_tokenize(sentence)
        finder = TrigramCollocationFinder.from_words(sentence_words)
        # sentence_trigrams = [trigram for (trigram, score) in finder.score_ngrams(self.trigram_measures.pmi)]
        sentence_trigrams = [trigram for (trigram, score) in finder.score_ngrams(self.trigram_measures.likelihood_ratio)]

        for trigram in self.trigram_features:
            features[trigram] = (trigram in sentence_trigrams)

        for word in self.word_features:
            features[word] = (word in sentence_words)

        return features

    def train(self):
        train_set = [(self.document_features(pos['text']), 'pos') for pos in POSYML]
        train_set += [(self.document_features(neg['text']), 'neg') for neg in NEGYML]

        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def save(self):
        # save classifier
        with open(CLASSIFIER_FILE,'wb') as file:
            joblib.dump(self.classifier, file)

    def load(self):
        # load classifier
        with open(CLASSIFIER_FILE, 'rb') as file:
            if file.read(1) != '':
                self.classifier = joblib.load(file)


    def parse(self, sentence):
        return self.classifier.classify(self.document_features(sentence))

    def accuracy(self):
        train_set = [(self.document_features(pos['text']), 'pos') for pos in POSYML]
        train_set += [(self.document_features(neg['text']), 'neg') for neg in NEGYML]

        return nltk.classify.accuracy(self.classifier, train_set)


n = NBClassifier()
print n.accuracy()
