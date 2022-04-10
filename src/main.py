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

    MAX_VECTORIZER_FEATURES = 100_000
    MAX_MODEL_ITER = 2_000

    logistic_regression_model = LogisticRegressionModel(out_path,
                                                        max_vectorizer_features=MAX_VECTORIZER_FEATURES,
                                                        max_model_iter=MAX_MODEL_ITER)
    logistic_regression_model.run()


def sentiment_analysis():
    # RNN-GRU modeli kullanarak duygu analizi.

    TRAINING_RATIO = 0.7
    ANALYSIS_DATA_PATH = "../data/sentiment_analysis_data.csv"

    rnn_gru_model = RnnGruModel(data_path=ANALYSIS_DATA_PATH,
                                training_ratio=TRAINING_RATIO)
    rnn_gru_model.run()


def main():
    # Parametreler
    ENCODING = 'utf-8'
    META_CHARACTERS = ["rt ", "\n", "\t"]
    COLUMN_INDEX = 8
    DATA_PATH = "../data/data.csv"
    OUT_PATH = "../data/preprocessed_data.csv"

    # preprocessing(DATA_PATH, ENCODING, META_CHARACTERS, COLUMN_INDEX, OUT_PATH)

    word_embedding(ENCODING, OUT_PATH)

    logistic_regression(OUT_PATH)

    sentiment_analysis()


if __name__ == "__main__":
    main()
