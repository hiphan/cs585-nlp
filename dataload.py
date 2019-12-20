import os
import codecs
import math
import numpy as np
from collections import Counter
import random
import pandas
import matplotlib.pyplot as plt

levels = ["0", "1", "2", "3", "4"]
train_data = []
test_data = []
counts = {}
total_words = {}

"""
This file contains methods for analyzing the Newsela dataset.
"""

# Load data from csv file (we used consistent train/test splits across experiments).
def load_csv_data():
    train_data = pandas.read_csv("training_newsela.csv")
    test_data = pandas.read_csv("test_newsela.csv")
    return train_data, test_data

# Calculate the maximum document length.
def max_doc_length():
    comb = pandas.concat([train_data, test_data])
    best = comb.iloc[0]
    for i, row in comb.iterrows():
        if len(row['text'].split()) > len(best['text'].split()):
            best = row
    print(best['text'])
    print("Longest document length: " + str(len(best['text'].split())))

# Create a histogram of document lengths.
def data_word_distribution():
    counts = []
    comb = pandas.concat([train_data, test_data])
    for i, row in comb.iterrows():
        counts.append(len(row['text'].split()))
    plt.hist(counts, bins=1000)
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Documents')
    plt.show()
    print(np.mean(counts))

train_data, test_data = load_csv_data()

# These are the two documents referenced in the paper.
# print(train_data.iloc[8])
# print(train_data.iloc[10]['text'])

# max_doc_length()

data_word_distribution()

"""

Max word count: 4502

"""