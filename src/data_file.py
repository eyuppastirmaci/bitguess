import pandas as pd
from preprocess import PreProcess


class DataFile:
    """
    Veri dosyasını işleyen sınıf.
    """

    def __init__(self, path: str, pre_process: PreProcess, out_path: str, encoding: str):
        """
        Yapıcı meteot.
        :param path: Dosyanın yolu.
        :param pre_process: Ön işleme nesnesi.
        """
        self._data_frame = pd.read_csv(path, encoding=encoding, low_memory=False, dtype='unicode')
        self._preprocess = pre_process
        self._out_path = out_path
        self._encoding = encoding
        self._corpus = []

    def pre_process_column(self, replaced_column_index: int):
        """
        Veri dosyasında ön işleme yapıp yeni dosya olarak kaydeden metot.
        """
        new_tweet_list = self._data_frame.text.tolist()
        for i in range(len(new_tweet_list)):
            new_tweet_list[i] = (self._preprocess.process(new_tweet_list[i]))
        column_number = self._data_frame.columns[replaced_column_index]
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
        self._data_frame.to_csv(self._out_path, encoding=self._encoding)
        print("Ön İşleme Tamamlandı!")
