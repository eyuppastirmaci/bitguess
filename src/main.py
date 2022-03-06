from process import TweetPreProcess
from data_file import CsvFile


def main():
    pre_process = TweetPreProcess()
    csv_file = CsvFile("data/turkish_tweets.csv")
    tweet_list = csv_file.data_frame.text.tolist()

    for tweet in tweet_list:
        print(pre_process.process(tweet))


if __name__ == '__main__':
    main()
