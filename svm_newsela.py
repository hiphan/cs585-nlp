import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from utils import *


class SVMClassifier:

    def __init__(self,
                 train_path='./data/training_newsela.csv',
                 test_path='./data/test_newsela.csv',
                 embedding_dim=300,
                 embedding_path='./glove.6B/glove.6B.300d.txt'):
        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.embedddings = self._get_embeddings()

        self.X_train, self.y_train = self._get_train(train_path)
        self.X_test, self.y_test = self._get_test(test_path)

    def _get_embeddings(self):
        """
        Load the whole embedding into memory
        :return:
        """
        embeddings_index = dict()
        f = open(self.embedding_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def _get_train(self, path):
        train_set = pd.read_csv(path)
        X_train_text = train_set['text']
        y_train = train_set['label']
        X_train = np.zeros((X_train_text.shape[0], self.embedding_dim))

        for i in range(X_train_text.shape[0]):
            text = X_train_text[i]
            curr_sum = np.zeros((self.embedding_dim,), dtype='float32')
            word_count = 0
            tokenized_text = tokenizer(text)
            for w in tokenized_text:
                if w in self.embedddings:
                    curr_sum += self.embedddings[w]
                    word_count += 1
            cur_average = curr_sum / word_count

            X_train[i, :] = cur_average

        return X_train, y_train

    def _get_test(self, path):
        test_set = pd.read_csv(path)
        X_test_text = test_set['text']
        y_test = test_set['label']
        X_test = np.zeros((X_test_text.shape[0], self.embedding_dim))

        for i in range(X_test_text.shape[0]):
            text = X_test_text[i]
            curr_sum = np.zeros((self.embedding_dim,), dtype='float32')
            word_count = 0
            tokenized_text = tokenizer(text)
            for w in tokenized_text:
                if w in self.embedddings:
                    curr_sum += self.embedddings[w]
                    word_count += 1
            cur_average = curr_sum / word_count

            X_test[i, :] = cur_average

        return X_test, y_test

    def train(self):
        Cs = np.arange(250, 2001, 250)

        best_acc, best_C = -1, None
        accuracies = []

        for C in Cs:
            total_acc = 0.0
            skf = StratifiedKFold(n_splits=5)
            for train_index, val_index in skf.split(self.X_train, self.y_train):
                X_train, X_val = self.X_train[train_index], self.X_train[val_index]
                y_train, y_val = self.y_train[train_index], self.y_train[val_index]
                clf = SVC(C=C, gamma='auto')
                clf.fit(X_train, y_train)
                split_acc = clf.score(X_val, y_val)
                total_acc += split_acc
            average_acc = total_acc / 5
            accuracies.append(average_acc)
            print('C: ' + str(C) + ', average accuracy: ' + str(average_acc))
            if average_acc > best_acc:
                best_acc = average_acc
                best_C = C

        clf = SVC(C=best_C, gamma='auto')
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        acc = np.mean(self.y_test == predictions)
        mae = np.mean(np.abs(self.y_test - predictions))
        mse = np.mean(np.square(self.y_test - predictions))
        print('Metrics with C = {}: {} accuracy, {} mae, {} mse'.format(best_C, acc, mae, mse))
        return (acc, mae, mse), Cs, accuracies


if __name__ == '__main__':
    SVM = SVMClassifier()
    metrics, Cs, accuracies = SVM.train()
    plt.plot(Cs, accuracies)
    plt.xlabel('Values of C')
    plt.ylabel('Accuracy')
    plt.savefig('svm_acc_newsela.png')

