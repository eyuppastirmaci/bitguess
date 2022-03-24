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

    def __update_column(self, index, new_tweet_list, out_path):
        """
        Csv dosyasının belirtilen indeksinde güncelleme yapan metot.
        """
        column_number = self._data_frame.columns[index]
        self._data_frame[column_number] = new_tweet_list
        self._data_frame.to_csv(out_path, encoding=self._encoding)

    def pre_process_column(self, index: int, out_path: str):
        """
        Veri dosyasında ön işleme yapıp yeni dosya olarak kaydeden metot.
        """
        new_tweet_list = self._preprocess.process(self.tweet_list)
        # Ön islenen sütun hariç diğer sütunların çıkarılması.
        for i in range(len(self._data_frame.columns)):
            if i != index:
                self._data_frame.drop(self._data_frame.columns[i])
        self.__update_column(index, new_tweet_list, out_path)
        print("--- Ön İşleme Tamamlandı ✓ ---")

    def extract_roots(self, index: int, out_path: str):
        """
        Kelimelerin Köklerini bularak yeni bir csv dosyası olarak kaydeden metot.
        """
        self.__update_column(index, self._preprocess.get_stem_words(self.tweet_list), out_path)

    def fix_typos(self, index: int, out_path: str):
        """
        Yazım yanlışlarını düzelterek yeni bir csv dosyası olarak kaydeden metot.
        """
        self.__update_column(index, self._preprocess.fix_typos(self.tweet_list), out_path)


