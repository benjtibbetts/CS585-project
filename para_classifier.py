from __future__ import division
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import csv
import re
import numpy as np
import random
import time
import math
from matplotlib import pyplot as plt
from collections import defaultdict
from collections import Counter
from sklearn import svm

para_label = ["(3, 2)","(4, 1)","(5, 0)"]
nonpara_label = ["(1, 4)", "(0, 5)"]
N_V_tags = ["nn", "nns", "vb", "vbd", "vbg", "vbn", "vbp", "vbz"]



"""
Takes either training or test data
Gets relevant information from data, calls fill_featlist for each pair
Also adds corresponding label of each pair of tweet to target list, 
used for y value of classifier.
"""
def construct_dataset(train=True):
	features = []
	target = []

	if train == True:
		path = "train.data"
	else:
		path = "test.data"

	data = []

	with open(path, 'rb') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter='\t')
	    for row in csv_reader:
	    	d = {}
	    	topic = row[1]
	    	sent1 = row[2]
	    	sent2 = row[3]
	    	label = row[4]
	    	sent1_tag = row[5]
	    	sent2_tag = row[6]

	    	if train == True:
		    	if label in para_label:
		    		target.append(1)
		    		fill_featlist(features, sent1, sent2, sent1_tag, sent2_tag, label, topic)

		    	if label in nonpara_label:
		    		target.append(0)
		    		fill_featlist(features, sent1, sent2, sent1_tag, sent2_tag, label, topic)

	    	if train == False:
	    		if int(label) > 3:
	    			target.append(1)
	    			fill_featlist(features, sent1, sent2, sent1_tag, sent2_tag, label, topic)

	    		if int(label) < 3:
	    			target.append(0)
	    			fill_featlist(features, sent1, sent2, sent1_tag, sent2_tag, label, topic)

	target = np.array(target)
	return features, target

"""
Creates a feature vector for pairs of tweets (in a dictionary)
Removes topic from both sentences
Splits irrelevant info from POS tags
Adds feature vector to a global list of features.
"""
def fill_featlist(features, sent1, sent2, sent1_tag, sent2_tag, label, topic):
	feat = {}

	sent1_no_topic = removeTopic(topic, sent1)
	sent2_no_topic = removeTopic(topic, sent2)

	sent1_list = tokenize(sent1_no_topic)
	sent2_list = tokenize(sent2_no_topic)
	
	sent1_pos = getPOS(tokenize(sent1_tag))
	sent2_pos = getPOS(tokenize(sent2_tag))

	stemmed_list1 = stemmer(sent1_list, sent1_pos)
	stemmed_list2 = stemmer(sent1_list, sent1_pos)

	feat["sent_cos"] = get_cosine(sent1_list, sent2_list)
	feat["pos_cos"] = get_cosine(sent1_pos, sent2_pos)
	feat["stem_cos"] = get_cosine(stemmed_list1, stemmed_list2)
	feat["len_difference"] = abs(len(sent1_list) - len(sent2_list))

	features.append(feat)

"""
Gets the POS tag for each word in a tokenized sentence.
"""
def getPOS(sent_tag):
	pos = []
	for t in sent_tag:
		tag_info = t.split("/")
		pos.append(tag_info[2])
	return pos

"removes topic from a sentence"
def removeTopic(topic, sent):
	regex = re.compile(r'\b('+topic+r')\b', flags=re.IGNORECASE)
	out = regex.sub("", sent)
	return out

"""
Takes a sentence and its POS tags and only stems the nouns and verbs
"""

st = LancasterStemmer()

def stemmer(sent, POS):
	for i in range(len(sent)):
		if POS[i] in N_V_tags:
			sent[i] = st.stem(sent[i])

	return sent



"""
Calculates the cosine similarity of words of two sentences. 
Also used to compare the sentences' POS tags.
"""
def get_cosine(sent1, sent2):
	vec1 = Counter(sent1)
	vec2 = Counter(sent2)

	intersection = set(vec1.keys()) & set(vec2.keys())
	numerator = sum([vec1[x] * vec2[x] for x in intersection])

	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	denominator = math.sqrt(sum1) * math.sqrt(sum2)

	if not denominator:
		return 0.0
	else:
		return float(numerator) / denominator

def tokenize(doc):
    tokens = doc.split()
    lowered_tokens = [t.lower() for t in tokens]
    return lowered_tokens

def most_informative_feature_for_class_svm(vectorizer, classifier,  n=10):
    labelid = 0 # this is the coef we're interested in. 
    feature_names = vectorizer.get_feature_names()
    svm_coef = classifier.coef_.toarray() 
    topn = sorted(zip(svm_coef[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print feat, coef

feats, target = construct_dataset()
vec = DictVectorizer()
X = vec.fit_transform(feats)

"""
The following is mostly a test to make sure dataset is formatted correctly.

Gaussian Naive Bayes prediction of a paraphrase using only sentential cosine distance 
and cosine distance of part of speech tags, only on test data.

"""

gnb = GaussianNB()
y_pred = gnb.fit(X.toarray(), target).predict(X.toarray())

print("Number of mislabeled points out of a total %d points : %d"
	% (X.shape[0],(target != y_pred).sum()))



"""
Evaluation of performance on test set using MultinomialNB
"""
test_feats, test_target = construct_dataset(train=False)

clf = MultinomialNB().fit(X, target)
test_vec = DictVectorizer()
test_X = test_vec.fit_transform(test_feats)

predicted = clf.predict(test_X)
print np.mean(predicted == test_target)

"""
Evaluation of performance on test set using linear SVM
"""
feats, target = construct_dataset()
vec = DictVectorizer()
X = vec.fit_transform(feats)

test_feats, test_target = construct_dataset(train=False)
test_vec = DictVectorizer()
test_X = test_vec.fit_transform(test_feats)

svm = svm.SVC(kernel='linear').fit(X, target)
predicted = svm.predict(test_X)
print np.mean(predicted == test_target)
most_informative_feature_for_class_svm(vec, svm)



"""
def make_feat_vec(sent1, sent2, sent1_tag, sent2_tag, label):
	feat_vec = defaultdict(float)
	for n in range(len(sent1) - 1):
		for i in range(len(sent2) - 1):
			feat_vec["str_%s_%s_%s" % (label, sent1[n], sent2[i])] = string_features(sent1[n], sent2[i])
			feat_vec["pos_%s_%s_%s" % (label, sent1[n], sent2[i])] = pos_features(getPOS(sent1_tag[n]), getPOS(sent2_tag[i]))

	return feat_vec



def create_sims(data_dict):
	documents = []
	documents.append(data_dict["sent1"])
	stoplist = set('for a of the and to in'.split())
	texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1

	texts = [[token for token in tex if frequency[token] > 1] for text in texts]
	dictionary = corpora.Dictionary(texts)
	new_vec = dictionary.doc2bow(data_dict["sent2"].lower().split())
	print new_vec
	corpus = [dictionary.doc2bow(text) for text in texts]
	tfidf = models.TfidfModel(corpus)
	print tfidf[new_vec]

data = construct_dataset()
create_sims(data[0])
dict_vectorizer = DictVectorizer()
vectorized = dict_vectorizer.fit_transform(data)


def create_vocab(train=True):
	if train == True:
		path = "train.data"
	else:
		path = "test.data"
	vocab = []
	with open(path, 'rb') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter='\t')
	    for row in csv_reader:
	    	vocab.append(row[2])
	    	vocab.append(row[3])
	return vocab

vectorizer = CountVectorizer()
vocab = create_vocab()
X = vectorizer.fit_transform(vocab)
counts = X.toarray()

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(counts)
print tfidf

def get_tfidf(sent1, sent2):
	vocab = [sent1, sent2]
	vectorizer = CountVectorizer(binary=True)
	X = vectorizer.fit_transform(vocab)
	counts = X.toarray()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(counts)
	return tfidf.toarray()


"""