import os
import codecs
import math
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression

levels = ["0", "1", "2", "3", "4"]
train_data = []
test_data = []
counts = {}
total_words = {}
vocab = []

def load_data():
	
	global train_data
	
	# Load data
	path = "articles"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			level = filename[-5:-4]
			if level == "5":
				continue
			text = file.read()
			train_data.append((text, level))
	
	train_data = train_data[0:1000]
	print("data loaded")

def split_data():
	global test_data
	global train_data
	#random.shuffle(train_data)
	n_test_data = math.ceil(len(train_data)*0.25)
	test_data = train_data[0:n_test_data]
	train_data = train_data[n_test_data+1:]

def get_vocabulary():
	for document in train_data:
		for word in document[0].split():
			if word not in vocab:
				vocab.append(word)

def get_document_vector(text):
	counts = []
	for word in vocab:
		counts.append(text.count(word))
	return counts

def get_vectorized_data(x):
	x_vec = []
	y_vec = []
	for doc in x:
		x_vec.append(get_document_vector(doc[0]))
		y_vec.append(int(doc[1]))
		
	assert len(x_vec) == len(y_vec)
	return x_vec, y_vec

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
			print(test_data[i][0])
			print(test_data[i][1])
		i = i + 1
	return np.mean(correct)

load_data()
#np.random.shuffle(train_data)
split_data()
get_vocabulary()
print(len(vocab))
print(logreg(.9))
