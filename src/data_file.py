import pandas as pd
from process import PreProcess


class CsvFile:
    """
    Csv dosyasını işleyen sınıf.
    """

    def __init__(self, path: str, pre_process: PreProcess):
        """
        Yapıcı meteot.
        :param path: Dosyanın yolu.
        :param use_column_names: Dosyadaki sütun isimleri.
        """
        pd.set_option('display.max_columns', 500)
        self.data_frame = pd.read_csv(path)
        self.pre_process = pre_process


    def pre_process_column(self, csv_data_frame, new_tweet_list: list, replaced_column_index: int):
        """
        Csv dosyasında ön işleme yapıp yeni dosya olarak kaydeden metot.
        :return:
        """
        for i in range(len(new_tweet_list)):
            new_tweet_list[i] = (self.pre_process.process(new_tweet_list[i]))
        column_number = csv_data_frame.columns[replaced_column_index]
        csv_data_frame.drop(column_number, axis=1, inplace=True)
        csv_data_frame[column_number] = new_tweet_list
        csv_data_frame.to_csv("data/pre_processed-turkish_tweets.csv", encoding='utf-8')
        print("Ön İşleme Tamamlandı!")
