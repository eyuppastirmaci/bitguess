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
        self.data_frame = pd.read_csv(path)
        self.preprocess = pre_process
        self.out_path = out_path
        self.encoding = encoding

    def pre_process_column(self, data_frame, new_tweet_list: list, replaced_column_index: int):
        """
        Veri dosyasında ön işleme yapıp yeni dosya olarak kaydeden metot.
        :return:
        """
        for i in range(len(new_tweet_list)):
            new_tweet_list[i] = (self.preprocess.process(new_tweet_list[i]))
        column_number = data_frame.columns[replaced_column_index]
        data_frame.drop(column_number, axis=1, inplace=True)
        data_frame[column_number] = new_tweet_list
        data_frame.to_csv(self.out_path, encoding=self.encoding)
        print("Ön İşleme Tamamlandı!")
