from preprocess import TweetPreProcess
from data_file import DataFile


def main():
    meta_characters = ["rt ", "\n", "\t"]

    column_index = 8

    data_path = "data/data.csv"
    out_path = "data/preprocessed-data.csv"
    encoding = 'utf-8'

    preprocess = TweetPreProcess(meta_characters)
    data_file = DataFile(data_path, preprocess, out_path, encoding)
    data_frame = data_file.data_frame
    data_list = data_frame.text.tolist()

    data_file.pre_process_column(data_frame, data_list, column_index)

    corpus = data_file.column_to_corpus("data/preprocessed-data.csv")



if __name__ == "__main__":
    main()
