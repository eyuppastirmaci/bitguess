from preprocess import TweetPreProcess
from data_file import DataFile
from word_embedding import WordVector


def main():
    # Parametreler
    meta_characters = ["rt ", "\n", "\t"]

    column_index = 8

    data_path = "data/data.csv"
    out_path = "data/preprocessed-data.csv"
    encoding = 'utf-8'

    # Ön işleme
    preprocess = TweetPreProcess(meta_characters)
    data_file = DataFile(data_path, preprocess, out_path, encoding)
    data_file.pre_process_column(column_index)

    # Kelime gömxme
    word_vector = WordVector("data/preprocessed-data.csv", encoding)
    model_sg = word_vector.model_sg
    model_cbow = word_vector.model_cbow


if __name__ == "__main__":
    main()
