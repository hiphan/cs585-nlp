import os
import codecs
import math
from collections import Counter
import random

levels = ["ele", "int", "adv"]
train_data = []
test_data = []
counts = {}
total_words = {}

def load_data():
	
	# Load elementary data
	path = "OneStopEnglishCorpus\Texts-SeparatedByReadingLevel\Ele-Txt"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			text = file.read()
			train_data.append((text, "ele"))
	
	# Load intermediate data
	path = "OneStopEnglishCorpus\Texts-SeparatedByReadingLevel\Int-Txt"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			text = file.read()
			train_data.append((text, "int"))
			
	# Load advanced data
	path = "OneStopEnglishCorpus\Texts-SeparatedByReadingLevel\Adv-Txt"
	for filename in os.listdir(path):
		if filename.endswith(".txt"):
			file = open(os.path.join(path, filename), encoding="utf-8")
			text = file.read()
			train_data.append((text, "adv"))


def split_data():
	global test_data
	global train_data
	
	n_test_data = math.ceil(len(train_data)*0.1)
	test_data = train_data[0:n_test_data]
	train_data = train_data[n_test_data+1:]

def train():
	counts['ele'] = Counter()
	counts['int'] = Counter()
	counts['adv'] = Counter()
	total_words['ele'] = 0
	total_words['int'] = 0
	total_words['adv'] = 0
	for text, level in train_data:
		for word in text.split():
			counts[level][word] += 1
			total_words[level] += 1

def classify(text):
	log_probs = {"ele":0, "int":0, "adv":0}
	for word in text.split():
		for level in levels:
			log_probs[level] += math.log((counts[level][word] + 1)/(total_words[level] + 1))
	return max(log_probs, key=log_probs.get)

def test():
	n_correct = 0
	for text, level in test_data:
		if classify(text) == level:
			n_correct += 1
		else:
			print("actual: ", level, " predicted: ", classify(text))
	accuracy = n_correct / len(test_data)
	print("Test Accuracy: ", accuracy)

load_data()
split_data()
train()
test()

