import pandas as pd
from preprocess import PreProcess


class DataFile:
    """
    Veri dosyasını işleyen sınıf.
    """

    def __init__(self, path: str, pre_process: PreProcess, encoding: str):
        """
        Yapıcı meteot.
        """
        self._data_frame = pd.read_csv(path, encoding=encoding, low_memory=False, dtype='unicode')
        self.tweet_list = self._data_frame.text.tolist()
        self._preprocess = pre_process
        self._encoding = encoding
        self._corpus = []

    def pre_process_column(self, index: int, out_path: str):
        """
        Veri dosyasında ön işleme yapıp yeni dosya olarak kaydeden metot.
        """
        new_tweet_list = self._data_frame.text.tolist()
        for i in range(len(new_tweet_list)):
            new_tweet_list[i] = (self._preprocess.process(new_tweet_list[i]))
        column_number = self._data_frame.columns[index]
        self._data_frame.drop("id", axis=1, inplace=True)
        self._data_frame.drop("user", axis=1, inplace=True)
        self._data_frame.drop("fullname", axis=1, inplace=True)
        self._data_frame.drop("url", axis=1, inplace=True)
        self._data_frame.drop("timestamp", axis=1, inplace=True)
        self._data_frame.drop("replies", axis=1, inplace=True)
        self._data_frame.drop("likes", axis=1, inplace=True)
        self._data_frame.drop("retweets", axis=1, inplace=True)
        self._data_frame.drop("sentiment", axis=1, inplace=True)
        self._data_frame[column_number] = new_tweet_list
        self._data_frame.to_csv(out_path, encoding=self._encoding)
        print("Ön İşleme Tamamlandı!")

    def stemming(self, index: int, out_path: str):
        """
        Kelimelerin Köklerini bularak yeni bir csv dosyası olarak kaydeden metot.
        """
        self.update_column(index, self._preprocess.get_stem_words(self.tweet_list), out_path)

    def typo_fixing(self, index: int, out_path: str):
        """
        Yazım yanlışlarını düzelterek yeni bir csv dosyası olarak kaydeden metot.
        """
        self.update_column(index, self._preprocess.fix_typos(self.tweet_list), out_path)

    def update_column(self, index, new_tweet_list, out_path):
        """
        Csv dosyasının belirtilen indeksinde güncelleme yapan metot.
        """
        column_number = self._data_frame.columns[index]
        self._data_frame[column_number] = new_tweet_list
        self._data_frame.to_csv(out_path, encoding=self._encoding)
