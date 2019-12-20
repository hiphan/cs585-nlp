import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import codecs


def create_data_csv(data_path='./newsela_corpus/newsela_article_corpus/articles'):
    print('There are %d examples in the Newsela dataset.' % len(os.listdir(data_path)))

    labels = [0, 1, 2, 3, 4]
    X, y = [], []
    for file in os.listdir(data_path):
        s = file.split('.')
        if s[1] == 'en':
            label = int(s[2])
            if label not in labels:
                continue
            with codecs.open(os.path.join(data_path, file), 'r', encoding='utf-8') as fp:
                text = fp.read()
            X.append(text)
            y.append(label)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        shuffle=True, stratify=y)

    train = np.concatenate((X_train, y_train), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_df = pd.DataFrame(data=train,
                            columns=['text', 'label'])
    test_df = pd.DataFrame(data=test,
                           columns=['text', 'label'])

    train_df.to_csv('data/training_newsela.csv')
    test_df.to_csv('data/test_newsela.csv')


if __name__ == '__main__':
    create_data_csv()
