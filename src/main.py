from process import TweetPreProcess
from data_file import CsvFile


def main():
    meta_characters = ["rt", "\n", "\t"]

    pre_process = TweetPreProcess(meta_characters)
    csv_file = CsvFile("data/turkish_tweets.csv", pre_process)
    csv_data_frame = csv_file.data_frame
    tweet_list = csv_data_frame.text.tolist()
    csv_file.pre_process_column(csv_data_frame, tweet_list, 8)


if __name__ == '__main__':
    main()
