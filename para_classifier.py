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
import nltk
from matplotlib import pyplot as plt
from collections import defaultdict
from collections import Counter
from sklearn import svm
from gensim import corpora, models, similarities
from collections import defaultdict
from nltk import Tree
from nltk.stem import WordNetLemmatizer

para_label = ["(3, 2)","(4, 1)","(5, 0)"]
nonpara_label = ["(1, 4)", "(0, 5)"]
N_tags = ["nn", "nns", "nnp", "nnps", "prp"]
V_tags = ["vb", "vbd", "vbg", "vbn", "vbp", "vbz", "md"]

bad_POS = ["uh", "sym", "rp", "fw", "ls", "wrb", "$", "pdt", 
"ht", "url", "pos", ".", ":", "rt", "usr", "``", "o", ",", "none", "''", ")", "#", "("]

test = []


# change this to True if you want to get results for specific features
# when running evaluate classifier
sent = True
pos = False
stem = False
length = False
n_lemma = False
v_lemma = True


"""
Takes either training or test data
Gets relevant information from data, calls fill_featlist for each pair
Also adds corresponding label of each pair of tweet to target list, 
used for y value of classifier.
"""
def construct_dataset(train=True):
	target = []
	features = []

	if train == True:
		path = "train.data"
	else:
		path = "test.data"

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
	    	if train==False:
	    		test.append((sent1, sent2, label))


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

	sent1_lemmatized_nouns = []
	sent1_lemmatized_verbs = []
	sent2_lemmatized_verbs = []
	sent2_lemmatized_nouns = []

	sent1_no_topic = removeTopic(topic, sent1)
	sent2_no_topic = removeTopic(topic, sent2)

	sent1_list = tokenize(sent1_no_topic)
	sent2_list = tokenize(sent2_no_topic)
	
	sent1_pos = getPOS(tokenize(sent1_tag))
	sent2_pos = getPOS(tokenize(sent2_tag))


	pos_for_tree1 = process_POS(sent1_pos)
	pos_for_tree2 = process_POS(sent2_pos)

	grammar = nltk.CFG.fromstring("""
		S -> NP VP | EX VP
		S -> S CC S
		CP -> P S 
		VP -> Vbar
		Vbar -> AdvP Vbar | Vbar PP | Vbar AdvP | V NP | V CP | V | TO Vbar | MD Vbar 
		NP -> D Nbar | Nbar
		Nbar -> AdjP Nbar | Nbar PP | N PP | N | N N
		PP -> AdjP Pbar | Pbar PP | P NP
		AdjP -> AdvP Adjbar | Adjbar PP | Adjbar | Adj PP
		N -> "nn" | "nns" | "nnp" | "nnps" | "wp" | "prp" | "wp" 
		EX -> "ex"
		V -> "vb" | "vbd" | "vbn" | "vbp" | "vbz" | "vbg"
		Adj -> "jj" | "jjs" | "jjr" | "cd" | "prp$"
		Adv -> "rb" | "rbr" | "rbs"
		D -> "dt" | "wdt"
		P -> "in"
		TO -> "to"
		MD -> "md"
		CC -> "cc"

		""")
	'''

	rd_parser = nltk.ChartParser(grammar)

	sent1_trees = []
	sent2_trees = []

	for tree in rd_parser.parse(pos_for_tree1 ):
		if tree.pformat() is not None:
			flat = tree.pformat().split()
			joined_flat = ' '.join(flat)
			sent1_trees.append(joined_flat)

	for tree in rd_parser.parse(pos_for_tree2 ):
		if tree.pformat() is not None:
			flat = tree.pformat().split()
			joined_flat = ' '.join(flat)
			sent2_trees.append(joined_flat)

	if len(sent1_trees) > 0 and len(sent2_trees) > 0:
		good_tree1 = sent1_trees[0]
		good_tree2 = sent1_trees[0]
		sent1_tree_brackets = good_tree1.count(")")
		sent2_tree_brackets = good_tree2.count(")")
		print sent1_tree_brackets
	else:
		sent1_tree_brackets = 0
		sent2_tree_brackets = 0

	'''
	stemmed_list1 = stemmer(sent1_list, sent1_pos)
	stemmed_list2 = stemmer(sent1_list, sent1_pos)

	wordnet_lemmatizer = WordNetLemmatizer()

	for i in range(len(sent1_list)):
		if sent1_pos[i] in V_tags:
			sent1_lemmatized_verbs.append(wordnet_lemmatizer.lemmatize(sent1_list[i], pos='v'))
		if sent1_pos[i] in N_tags:
			sent1_lemmatized_nouns.append(wordnet_lemmatizer.lemmatize(sent1_list[i], pos='n'))

	for i in range(len(sent2_list)):
		if sent2_pos[i] in V_tags:
			sent2_lemmatized_verbs.append(wordnet_lemmatizer.lemmatize(sent2_list[i], pos='v'))
		if sent2_pos[i] in N_tags:
			sent2_lemmatized_nouns.append(wordnet_lemmatizer.lemmatize(sent2_list[i], pos='n'))

	sent1_lemmatized_nouns = set(sent1_lemmatized_nouns)
	sent2_lemmatized_nouns = set(sent2_lemmatized_nouns)
	sent1_lemmatized_verbs = set(sent1_lemmatized_verbs)
	sent2_lemmatized_verbs = set(sent2_lemmatized_verbs)

	if sent:
		feat["sent_cos"] = get_cosine(sent1_list, sent2_list)
	if pos:
		feat["pos_cos"] = get_cosine(sent1_pos, sent2_pos)
	if stem:
		feat["stem_cos"] = get_cosine(stemmed_list1, stemmed_list2)
	if length:
		feat["len_difference"] = abs(len(sent1_list) - len(sent2_list))
		# feat["syn_similarity"] = abs(sent1_tree_brackets - sent2_tree_brackets)
	if n_lemma:
		feat["lemmatized_noun_overlap"] = len(sent2_lemmatized_nouns.intersection(sent1_lemmatized_nouns))
	if v_lemma:
		feat["lemmatized_verb_overlap"] = len(sent2_lemmatized_verbs.intersection(sent1_lemmatized_verbs))


	features.append(feat)

"""
Takes a list of POS as input and removes the tags that our parser can't handle
such as interjections and wh-words.
"""
def process_POS(pos):
	new_list = []
	for p in pos:
		if p not in bad_POS:
			new_list.append(p)
	return new_list


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
		if POS[i] in N_tags or V_tags:
			sent[i] = st.stem(sent[i])

	return sent


wordnet_lemmatizer = WordNetLemmatizer()


"""
given a word and it's part of speech, lemmatize it
"""
def lemmatizer(word, pos):
	word = wordnet_lemmatizer.lemmatize(word,)


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


"""
The following is mostly a test to make sure dataset is formatted correctly.

Gaussian Naive Bayes prediction of a paraphrase using only sentential cosine distance 
and cosine distance of part of speech tags, only on test data.

"""
def evalueateGNB():
	feats, target = construct_dataset()
	vec = DictVectorizer()
	X = vec.fit_transform(feats)
	gnb = GaussianNB()
	y_pred = gnb.fit(X.toarray(), target).predict(X.toarray())

	print("Number of mislabeled points out of a total %d points : %d"
		% (X.shape[0],(target != y_pred).sum()))



"""
Evaluation of performance on test set using MultinomialNB
"""
def evaluateMNB():
	feats, target = construct_dataset()
	vec = DictVectorizer()
	X = vec.fit_transform(feats)
	test_feats, test_target = construct_dataset(train=False)

	clf = MultinomialNB().fit(X, target)
	test_vec = DictVectorizer()
	test_X = test_vec.fit_transform(test_feats)

	predicted = clf.predict(test_X)
	print np.mean(predicted == test_target)

"""
Evaluation of performance on test set using linear SVM

"""
def evaluateSVM():
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