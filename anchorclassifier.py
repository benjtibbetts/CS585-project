from __future__ import division
import csv
import numpy as np
import random
import time
import re
import math
from matplotlib import pyplot as plt
from collections import Counter
from collections import defaultdict


para_label = ["(3, 2)","(4, 1)","(5, 0)"]
nonpara_label = ["(1, 4)", "(0, 5)"]
WORD = re.compile(r'\w+')

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    tokens = doc.split()
    lowered_tokens = [t.lower() for t in tokens]
    return lowered_tokens

def getPOS(sent_tag):
	tokens = sent_tag.split("/")
	return tokens[2]

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: out[k] -= vec2[k]
    return dict(out)

def dict_argmax(dct):
    """Return the key whose value is largest. In other words: argmax_k dct[k]"""
    return max(dct.iterkeys(), key=lambda k: dct[k])

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.iterkeys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def text_to_vector(text):
	words = WORD.findall(text)
	return Counter(words)

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


def construct_dataset(train=True):
	if train == True:
		path = "train.data"
	else:
		path = "test.data"
	data = []
	with open(path, 'rb') as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter='\t')
	    for row in csv_reader:
	    	d = {}
	    	d["topic_id"] = row[0]
	    	d["topic_name"] = row[1]
	    	d["sent1"] = tokenize_doc(row[2]) 
	    	d["sent2"] = tokenize_doc(row[3])
	    	d["label"] = row[4]
	    	d["sent1_tag"] = tokenize_doc(row[5])
	    	d["sent2_tag"] = tokenize_doc(row[6])
	    	if d["label"] in para_label:
	    		data.append((d, 1))
	    	if d["label"] in nonpara_label:
	    		data.append((d, -1))
	return data


features = []

def make_feat_vec(sent1, sent2, sent1_tag, sent2_tag, label):
	feat_vec = defaultdict(float)
	pos1 = []
	pos2 = []

	for n in range(len(sent1_tag)):
		pos1.append(getPOS(sent1_tag[n]))

	for n in range(len(sent2_tag)):
		pos2.append(getPOS(sent2_tag[n]))

	for n in range(len(sent1) - 1):
		for i in range(len(sent2) - 1):
			feat_vec["str_%s_%s_%s" % (label, sent1[n], sent2[i])] = string_features(sent1[n], sent2[i])
			feat_vec["pos_%s_%s_%s" % (label, sent1[n], sent2[i])] = pos_features(getPOS(sent1_tag[n]), getPOS(sent2_tag[i]))
	feat_vec["sent_cosine_%s_%s_%s" % (label, sent1, sent2)] = get_cosine(sent1, sent2)
	feat_vec["POS_cosine_%s_%s_%s" % (label, sent1, sent2)] = get_cosine(pos1, pos2)

	return feat_vec



def predict_para(sent1, sent2, sent1_tag, sent2_tag, weights, para_feat_vec=None, nonpara_feat_vec=None):
	para_feat_vec = make_feat_vec(sent1, sent2, sent1_tag, sent2_tag, "1") if para_feat_vec == None else para_feat_vec
	nonpara_feat_vec = make_feat_vec(sent1, sent2, sent1_tag, sent2_tag, "-1") if nonpara_feat_vec == None else nonpara_feat_vec
	scores = { 1: dict_dotprod(para_feat_vec, weights),
				-1: dict_dotprod(nonpara_feat_vec, weights) }

	return dict_argmax(scores)


def train(examples, stepsize=1, numpasses=10, do_averaging=False, devdata=None):

	weights = defaultdict(float)
	weightSums = defaultdict(float)
	t = 0

	train_acc = []
	test_acc = []
	avg_test_acc = []

	def get_averaged_weights():
		return {f: weights[f] - 1/t*weightSums[f] for f in weightSums}

	for pass_iteration in range(numpasses):
		start = time.time()
		print "\tTraining iteration %d" % pass_iteration
		random.shuffle(examples)
		for d, goldlabel in examples:
			t += 1
			sent1 = d["sent1"]
			sent2 = d["sent2"]
			sent1_tag = d["sent1_tag"]
			sent2_tag = d["sent2_tag"]
			para_feat_vec = make_feat_vec(sent1, sent2, sent1_tag, sent2_tag, "1")
			nonpara_feat_vec = make_feat_vec(sent1, sent2, sent1_tag, sent2_tag, "-1")
			predlabel = predict_para(sent1, sent2, sent1_tag, sent2_tag, weights, para_feat_vec, nonpara_feat_vec)
			if predlabel != goldlabel:
				predfeats = para_feat_vec if goldlabel == -1 else nonpara_feat_vec
				goldfeats = para_feat_vec if goldlabel ==  1 else nonpara_feat_vec
				featdelta = dict_subtract(goldfeats, predfeats)
				for feat_name, feat_value in featdelta.iteritems():
					weights[feat_name] += stepsize * feat_value
					weightSums[feat_name] += (t-1) * stepsize * feat_value
		end = time.time()
		print end - start
		print "TR RAW EVAL:",
		train_acc.append(do_evaluation(examples, weights))

		if devdata:
			print "DEV RAW EVAL:",
			test_acc.append(do_evaluation(devdata, weights))

		if devdata and do_averaging:
			print "DEV AVG EVAL:",
			avg_test_acc.append(do_evaluation(devdata, get_averaged_weights()))

	print "[learned weights for %d features from %d examples.]" % (len(weights), len(examples))

	return { 'train_acc': train_acc,
	 'test_acc': test_acc,
	 'avg_test_acc': avg_test_acc,
	 'weights': weights if not do_averaging else get_averaged_weights() }


def string_features(word1, word2):
	if word1 == word2:
		return 1
	else:
		return 0

def pos_features(word1_tag, word2_tag):
	if word1_tag == word2_tag:
		return 1
	else:
		return 0
	
def do_evaluation(examples, weights):
    """
    Compute the accuracy of a trained perceptron.
    """
    num_correct, num_total = 0, 0
    for d, goldlabel in examples:
    	sent1 = d["sent1"]
    	sent2 = d["sent2"]
    	sent1_tag = d["sent1_tag"]
    	sent2_tag = d["sent2_tag"]
        predlabel = predict_para(sent1, sent2, sent1_tag, sent2_tag, weights)

        if predlabel == goldlabel:
            num_correct += 1.0
        num_total += 1.0
    print "%d/%d = %.4f accuracy" % (num_correct, num_total, num_correct/num_total)
    return num_correct/num_total

def plot_accuracy_vs_iteration(train_acc, test_acc, avg_test_acc, naive_bayes_acc = 0.83):
    """
    Plot the vanilla perceptron accuracy on the trainning set and test set
    and the averaged perceptron accuracy on the test set.
    """

    plt.plot(range(len(train_acc)), train_acc)
    plt.plot(range(len(test_acc)), test_acc)
    plt.plot(range(len(avg_test_acc)), avg_test_acc)
    plt.plot(range(len(avg_test_acc)), [naive_bayes_acc] * len(avg_test_acc))
    plt.xlabel('Num Training Iterations')
    plt.ylabel('Accuracy (%)')
    plt.title('Iterations  vs. Accuracy')
    plt.show()

if __name__=='__main__':
    training_set = construct_dataset(train=True)
    test_set = construct_dataset(train=False)
    # sol_dict = train(training_set, do_averaging=False, devdata=test_set)
    sol_dict = train(training_set, stepsize=1, numpasses=10, do_averaging=True, devdata=test_set)
    plot_accuracy_vs_iteration(sol_dict['train_acc'], sol_dict['test_acc'], sol_dict['avg_test_acc'])


