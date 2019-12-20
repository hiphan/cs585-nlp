import os
import codecs
import math
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression

"""
Runs logistic regression experiment on OneStopEnglish Corpus.
"""

# Defines the class that we use to represent all OneStopEnglishCorpus data.
class InputExample(object):
	"""A single training/test example for sequence classification."""

	def __init__(self, guid, text_a, text_b=None, labels=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			labels: (Optional) [string]. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.labels = labels

	def summary(self):
		return "[" + str(self.guid) + ": " + self.text_a[0:20] + "]"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

levels = ["ele", "int", "adv"]
train_data = []
test_data = []
counts = {}
total_words = {}
vocab = []

# Loads all OneStopEnglishCorpus data into array train_data.
def load_data():
	
	current_id = 0
	
	# Load elementary data
	path = "OneStopEnglishCorpus\Texts-SeparatedByReadingLevel\Ele-Txt"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			text = file.read()[1:]
			train_data.append(InputExample(current_id, text, labels=["ele"]))
			current_id += 1
	
	# Load intermediate data
	path = "OneStopEnglishCorpus\Texts-SeparatedByReadingLevel\Int-Txt"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			text = file.read()[14:]
			train_data.append(InputExample(current_id, text, labels=["int"]))
			current_id += 1
			
	# Load advanced data
	path = "OneStopEnglishCorpus\Texts-SeparatedByReadingLevel\Adv-Txt"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			text = file.read()[1:]
			train_data.append(InputExample(current_id, text, labels=["adv"]))
			current_id += 1
	
	print("Data loaded.")

# Splits the data in train_data into train and test data.
def split_data():
	global test_data
	global train_data
	n_test_data = math.ceil(len(train_data)*0.1)
	test_data = train_data[0:n_test_data]
	train_data = train_data[n_test_data+1:]

# Initializes the vocabulary.
def get_vocabulary():
	for document in train_data:
		for word in document.text_a.split():
			if word not in vocab:
				vocab.append(word)

# Given a document, makes a bag of words vector representing that document.
def get_document_vector(text):
	counts = []
	for word in vocab:
		counts.append(text.count(word))
	return counts

# Converts all the string documents into bag of words vectors.
def get_vectorized_data(x):
	x_vec = []
	y_vec = []
	for doc in x:
		x_vec.append(get_document_vector(doc.text_a))
		if doc.labels[0] == "ele":
			y_vec.append(0)
		elif doc.labels[0] == "int":
			y_vec.append(1)
		elif doc.labels[0] == "adv":
			y_vec.append(2)
		else:
			print("this is bad")
			return -1
	assert len(x_vec) == len(y_vec)
	return x_vec, y_vec

# Runs logistic regression.
def logreg(reg = .9):
	clf = LogisticRegression(C=reg)
	x, y = get_vectorized_data(train_data)
	clf.fit(x, y)
	test_x, test_y = get_vectorized_data(test_data)
	pred = clf.predict(test_x)
	correct = []
	i = 0
	for p, ty in zip(pred, test_y):
		correct.append(p == ty)
		if not p == ty:
			print(test_data[i].text_a)
			print(test_data[i].labels[0])
		i = i + 1
	return np.mean(correct)

# Calculates the max document length.
def max_doc_length():
	best = train_data[0]
	for doc in train_data:
		if len(doc.text_a) > len(best.text_a):
			best = doc
	print(doc.text_a)
	print(len(doc.text_a.split()))

load_data()
# We want the train/test split to be random.
np.random.shuffle(train_data)
split_data()
get_vocabulary()
print(len(vocab))
print(logreg(.9))