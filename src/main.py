from process import TweetPreProcess
from data_file import CsvFile


def main():
    meta_characters = ["rt ", "\n", "\t"]
    csv_file_path = "data/turkish_tweets.csv"
    column_index = 8

    pre_process = TweetPreProcess(meta_characters)
    csv_file = CsvFile(csv_file_path, pre_process)
    csv_data_frame = csv_file.data_frame
    tweet_list = csv_data_frame.text.tolist()
    csv_file.pre_process_column(csv_data_frame, tweet_list, column_index)


if __name__ == '__main__':
    main()
