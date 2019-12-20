import nltk
from nltk import word_tokenize
import string
import json


nltk.download('punkt')


def tokenizer(content):
    # To lowercase
    s = content.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    return word_tokenize(s)
