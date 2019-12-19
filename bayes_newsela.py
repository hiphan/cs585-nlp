import os
import codecs
import math
from collections import Counter
import random

levels = ["0", "1", "2", "3", "4"]
train_data = []
test_data = []
counts = {}
total_words = {}

def load_data():
	
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


def split_data():
	global test_data
	global train_data
	
	n_test_data = math.ceil(len(train_data)*0.25)
	test_data = train_data[0:n_test_data]
	train_data = train_data[n_test_data+1:]

def train():
	for level in levels:
		counts[level] = Counter()
		total_words[level] = 0
	for text, level in train_data:
		for word in text.split():
			counts[level][word] += 1
			total_words[level] += 1

def classify(text):
	log_probs = {}
	for level in levels:
		log_probs[level] = 0
	for word in text.split():
		for level in levels:
			log_probs[level] += math.log((counts[level][word] + 1)/(total_words[level] + 1))
	return max(log_probs, key=log_probs.get)

def test():
	n_correct = 0
	total_error = 0
	total_squared_error = 0
	for text, level in test_data:
		prediction = classify(text)
		error = abs(int(prediction) - int(level))
		total_error += error
		total_squared_error += error*error
		if prediction == level:
			n_correct += 1
		else:
			print("actual: ", level, " predicted: ", prediction)
	accuracy = n_correct / len(test_data)
	print("Test Accuracy: ", accuracy)
	print("Test MAE: ", total_error/len(test_data))
	print("Test MSE: ", total_squared_error/len(test_data))

load_data()
split_data()
train()
test()

