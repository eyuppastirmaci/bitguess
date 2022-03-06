import pandas as pd


class CsvFile:
    """
    Csv dosyasını işleyen sınıf.
    """

    def __init__(self, path: str):
        """
        Yapıcı meteot.
        :param path: Dosyanın yolu.
        :param use_column_names: Dosyadaki sütun isimleri.
        """
        pd.set_option('display.max_columns', 500)
        self.data_frame = pd.read_csv(path)

    def get_column_list(self):
        """
        Dosyadaki bir sütunu liste olarak döndüren metot.
        :return: list
        """
        return self.data_frame.to_list()



