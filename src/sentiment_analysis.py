import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LogisticRegressionModel:

    def __init__(self, path, max_vectorizer_features, max_model_iter):
        self.path = path
        self.vectorizer = TfidfVectorizer(max_features=max_vectorizer_features)
        self.model = LogisticRegression(max_iter=max_model_iter)

    def get_word_weights(self):
        df_ = pd.read_csv(self.path)[['sentiment', 'text']].copy()
        df = df_[['sentiment', 'text']].copy()
        target_map = {3: 3, 2: 2, 1: 1, 0: 0, -1: -1}
        df['target'] = df['sentiment'].map(target_map)
        df_train, df_test = train_test_split(df)
        binary_target_list = [target_map[1], target_map[-1]]
        df_b_train = df_train[df_train['target'].isin(binary_target_list)]
        X_train = self.vectorizer.fit_transform(df_b_train['text'])
        Y_train = df_b_train['target']
        self.model.fit(X_train, Y_train)
        word_index_map = self.vectorizer.vocabulary_

        print("En olumlu kelimeler")
        for word, index in word_index_map.items():
            weight = self.model.coef_[0][index]
            if weight > 2:
                print(word, weight)

        print("\n\n")

        print("En olumsuz kelimeler")
        for word, index in word_index_map.items():
            weight = self.model.coef_[0][index]
            if weight < -2:
                print(word, weight)

