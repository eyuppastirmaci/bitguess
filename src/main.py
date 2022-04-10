from preprocess import TweetPreProcess
from data_file import DataFile
from word_vector import WordVector
from analysis.regression import LogisticRegressionModel
from analysis.sentiment import RnnGruModel


def preprocessing(data_path, encoding, meta_characters, column_index, out_path):
    # Ön işleme
    preprocess = TweetPreProcess(meta_characters=meta_characters)
    data_file = DataFile(data_path, preprocess, encoding)
    data_file.pre_process_column(column_index, out_path)


def word_embedding(encoding, out_path):
    # Kelime gömme.
    word_vector = WordVector(out_path, encoding)
    word_vector.run()


def logistic_regression(out_path):
    # Lojistik Regresyon modeli kullanarak regresyon analizi.
    logistic_regression_model = LogisticRegressionModel(out_path, max_vectorizer_features=100000, max_model_iter=2000)
    logistic_regression_model.run()


def sentiment_analysis(sentiment_analysis_data_path, training_ratio):
    # RNN-GRU modeli kullanarak duygu analizi.
    rnn_gru_model = RnnGruModel(sentiment_analysis_data_path, training_ratio=training_ratio)
    rnn_gru_model.run()


def main():
    # Parametreler
    ENCODING = 'utf-8'
    META_CHARACTERS = ["rt ", "\n", "\t"]
    COLUMN_INDEX = 8
    DATA_PATH = "../data/data.csv"
    OUT_PATH = "../data/preprocessed_data.csv"
    SENTIMENT_ANALYSIS_DATA_PATH = "../data/sentiment_analysis_data.csv"
    TRAINING_RATIO = 0.7

    # preprocessing(DATA_PATH, ENCODING, META_CHARACTERS, COLUMN_INDEX, OUT_PATH)

    word_embedding(ENCODING, OUT_PATH)

    logistic_regression(OUT_PATH)

    sentiment_analysis(SENTIMENT_ANALYSIS_DATA_PATH, training_ratio=TRAINING_RATIO)


if __name__ == "__main__":
    main()
