import nltk
from nltk.corpus import sentence_polarity
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from unidecode import unidecode
import cPickle as pickle
import random

# inherit .classify() from ClassifierI
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


files = {"POSITIVE_FILENAME": "pos",
		 "NEGATIVE_FILENAME": "neg",
		 "positive.txt": "pos",
		 "negative.txt": "neg"}
		 

# this will be a list of tuples containing documents and categories
documents = []
allowed_words = []

allowed_word_types = ["J"]

# tokenize, tag, and append documents with their categories
def appending(docs, category):
    for doc in docs.split('\n'):
        # append tuples to the list of documents
	documents.append( (doc, category) )
	words = word_tokenize(doc)
	tagged = nltk.pos_tag(words) # pos_tag()returns a list of tuples
		
	# iterating over the list of tuples
	for w in tagged:
		# slicing the first letter of the second item in the tuple
		if w[1][0] in allowed_word_types:
			allowed_words.append(w[0].lower())


# open and decode each file in the files dictionary
for f in files.items():
    data = open(f[0], "r").read()
    docs = data.decode('unicode_escape').encode('ascii','ignore')
    appending(docs, f[1])


save_documents = open("documents.pickle", "wb+")
pickle.dump(documents, save_documents)
save_documents.close()


# order allowed_words by frequency distribution from most common to least common
allowed_words = nltk.FreqDist(allowed_words)

#create a list of of the most common words without the sentiment tag values
word_features = list(allowed_words.keys())[:8000]


save_word_features = open("word_features.pickle", "wb+")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(sentence):
    words = word_tokenize(sentence)
    features = {}
	
    for word in word_features:
        # update the features dict with {"word": boolean}
        features[word] = (word in words) # boolean, if word is in words, return True
    # return the dictionary of words as keys and booleans as values
    return features

# featuresets is a list of tuples each containing a document and a category
featuresets = [(find_features(sentence), category) for (sentence, category) in documents]
                # ^only gets the key from the features dictionary

save_featuresets = open("featuresets.pickle", "wb+")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

# shuffle positive and negative features
random.shuffle(featuresets)

training_set = featuresets[:18000]
testing_set = featuresets[18000:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print "Original Naive Bayes Algo accuracy: %" + str(nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(15)

save_classifier = open("originalnaivebayes.pickle", "wb+")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LinearSVC_classifier = SklearnClassifier(LinearSVC())
NuSVC_classifier = SklearnClassifier(NuSVC())
SGDC_classifier = SklearnClassifier(SGDClassifier())


algos = {"MultinomialNB_classifier": MultinomialNB_classifier,
	 "BernoulliNB_classifier": BernoulliNB_classifier,
	 "LogisticRegression_classifier": LogisticRegression_classifier,
	 "LinearSVC_classifier": LinearSVC_classifier,
	 "NuSVC_classifier": NuSVC_classifier,
	 "SGDC_classifier": SGDC_classifier}


for i in algos.items():
    i[1].train(training_set)
    print i[0], "accuracy: %" + str(nltk.classify.accuracy(i[1], testing_set) * 100)
	
    save_classifier = open(i[0]+".pickle", "wb+")
    pickle.dump(i[1], save_classifier)
    save_classifier.close()
	

# set voted_classifier to an instance of the class VoteClassifiers
voted_classifier = VoteClassifier(classifier, 
				  MultinomialNB_classifier, 
				  BernoulliNB_classifier, 
				  LogisticRegression_classifier, 
				  NuSVC_classifier,
				  LinearSVC_classifier,
				  SGDC_classifier)

print "voted_classifier accuracy: %" + str(nltk.classify.accuracy(voted_classifier, testing_set) * 100)



