import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from utils import *


class SVMClassifier:

    def __init__(self,
                 data_path='./data/all.json',
                 embedding_dim=300,
                 embedding_path='./glove.6B/glove.6B.300d.txt'):
        self.data_path = data_path
        self.data_count = self._data_count()

        self.embedding_dim = embedding_dim
        self.embedding_path = embedding_path
        self.embedddings = self._get_embeddings()

        self.X, self.y = self._preprocess_data()

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

    def _data_count(self):
        with open(self.data_path, 'r') as fp:
            data = json.load(fp)

        count = 0
        for level in data:
            count += len(data[level])
        print('There are %d text documents.' % count)
        return count

    def _preprocess_data(self):
        X = np.zeros((self.data_count, self.embedding_dim))
        y = np.zeros((self.data_count, ))

        with open(self.data_path, 'r') as fp:
            data = json.load(fp)

        labels = {
            'elementary': 0,
            'intermediate': 1,
            'advanced': 2
        }

        idx = 0
        for level in data:
            for _, text in data[level].items():
                curr_sum = np.zeros((self.embedding_dim, ), dtype='float32')
                word_count = 0
                tokenized_text = tokenizer(text)
                for w in tokenized_text:
                    if w in self.embedddings:
                        curr_sum += self.embedddings[w]
                        word_count += 1
                cur_average = curr_sum / word_count

                X[idx, :] = cur_average
                y[idx] = labels[level]

                idx += 1

        return X, y

    def train(self):
        Cs = np.arange(250, 2001, 250)
        print(Cs)

        X_train_all, X_test, y_train_all, y_test = train_test_split(self.X, self.y, test_size=0.15, random_state=0)

        best_acc, best_C = -1, None
        accuracies = []

        for C in Cs:
            total_acc = 0.0
            skf = StratifiedKFold(n_splits=5)
            for train_index, val_index in skf.split(X_train_all, y_train_all):
                X_train, X_val = X_train_all[train_index], X_train_all[val_index]
                y_train, y_val = y_train_all[train_index], y_train_all[val_index]
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
        clf.fit(X_train_all, y_train_all)
        accuracy = clf.score(X_test, y_test)
        print('Test accuracy with C = {}: {}'.format(best_C, accuracy))
        return accuracy, Cs, accuracies


if __name__ == '__main__':
    SVM = SVMClassifier()
    accuracy, Cs, accuracies = SVM.train()
    plt.plot(Cs, accuracies)
    plt.xlabel('Values of C')
    plt.ylabel('Accuracy')
    plt.savefig('svm_acc.png')
