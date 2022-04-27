import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizer_v1 import Adam


class RnnGruModel:

    def __init__(self, data_path, training_ratio):
        self.path = data_path
        self.training_ratio = training_ratio

    def run(self):
        NUMBER_OF_WORDS = 15_000
        EMBEDDING_SIZE = 50

        tf.compat.v1.disable_eager_execution()

        dataframe = pd.read_csv(self.path)
        target = dataframe['sentiment'].values.tolist()
        data = dataframe['text'].values.tolist()

        for i in range(len(data)):
            data[i] = str(data[i])

        cutoff = int(len(data) * self.training_ratio)
        x_train, x_test = data[:cutoff], data[cutoff:]
        y_train, y_test = target[:cutoff], target[cutoff:]

        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(data)
        x_train_tokens = tokenizer.texts_to_sequences(x_train)
        x_test_tokens = tokenizer.texts_to_sequences(x_test)

        num_tokens = np.array([len(tokens) for tokens in x_train_tokens + x_test_tokens])
        max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))

        x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_tokens, maxlen=max_tokens)
        x_test_pad = tf.keras.preprocessing.sequence.pad_sequences(x_test_tokens, maxlen=max_tokens)

        model = Sequential()
        model.add(Embedding(input_dim=NUMBER_OF_WORDS,
                            output_dim=EMBEDDING_SIZE,
                            input_length=max_tokens,
                            name="embedding_layer"))

        model.add(GRU(units=16, return_sequences=True))
        model.add(GRU(units=8, return_sequences=True))
        model.add(GRU(units=4))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(lr=1e-3)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.fit(x_train_pad, y_train, epochs=5, batch_size=256)

        result = model.evaluate(x_test_pad, y_test)

        y_pred = model.predict(x=x_test_pad[0:1200])
        y_pred = y_pred.T[0]

        cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
        cls_true = np.array(y_test[0:1200])

        correct = len(np.where(cls_pred == cls_true)[0])
        incorrect = len(np.where(cls_pred != cls_true)[0])

        print("\n\n======== Duygu Analizi ========")
        print(f"Doğru Tahmin: {correct}")
        print(f"Yanlış Tahmin: {incorrect}")
        print(f"Doğru Tahmin Yüzdesi: %{round(result[1]*100,2)}")
