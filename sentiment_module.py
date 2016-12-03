import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import cPickle as pickle


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from nltk.classify import ClassifierI
#from statistics import mode

from nltk.tokenize import word_tokenize
from unidecode import unidecode


class VoteClassifier(ClassifierI):
	
    def __init__(self, *classifiers):
        self._classifiers = classifiers
		
    def classify(self, features):
        votes = []
        for c in self._classifiers:
	    v = c.classify(features)
	    votes.append(v)
	return max(set(votes), key=votes.count) # mode without importing mode
		
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
	    v = c.classify(features)
	    votes.append(v)
		
	choice_votes = votes.count(max(set(votes), key=votes.count)) # mode
	conf = float(choice_votes) / len(votes)
        return conf

opins_sents_file = open("opins_sents.pickle", "rb")
opins_sents = pickle.load(opins_sents_file)
opins_sents_file.close()


word_features_file = open("word_features.pickle", "rb")
word_features = pickle.load(word_features_file)
word_features_file.close()

def find_features(opin):
    words = word_tokenize(opin)
    features = {}
	
    for w in word_features:
        features[w] = (w in words) # boolean, if w is in words, it will be True
	
    return features

featuresets_file = open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_file)
featuresets_file.close()

random.shuffle(featuresets)


open_file = open("originalnaivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("MultinomialNB_classifier.pickle", "rb")
MultinomialNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

# set voted_classifier to an instance of the class VoteClassifier 
voted_classifier = VoteClassifier(classifier, 
				  MultinomialNB_classifier, 
				  BernoulliNB_classifier, 
	 			  LogisticRegression_classifier, 
				  NuSVC_classifier,
				  LinearSVC_classifier,
				  SGDC_classifier)


def sentiment(text):
	feats = find_features(text)
	# from voted_classifier, get the classify function and call it with feats
	# also get the confidence function and call it with feats
	return voted_classifier.classify(feats), voted_classifier.confidence(feats)
	
