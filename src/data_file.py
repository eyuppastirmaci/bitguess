import pandas as pd


class CsvFile:
    """
    Csv dosyasını işleyen sınıf.
    """

    def __init__(self, path: str, column_names: list):
        """
        Yapıcı meteot.
        :param path: Dosyanın yolu.
        :param column_names: Dosyadaki sütun isimleri.
        """
        self.data_frame = pd.read_csv(path, column_names)

    def get_column_list(self):
        """
        Dosyadaki bir sütunu liste olarak döndüren metot.
        :return: list
        """
        return self.data_frame.to_list()



