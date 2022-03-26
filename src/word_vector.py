from gensim.models import Word2Vec
import pandas as pd


class WordVector:
    """
    Kelime vektör sınıfı.
    """

    def __init__(self, path: str, encoding: str):
        """
        Yapıcı metot.
        """
        self._data = pd.read_csv(path, encoding=encoding, low_memory=False).text.tolist()
        self._corpus = self.__get_corpus()
        self.model_sg = self.__get_model(1)
        self.model_cbow = self.__get_model(0)

    def __get_model(self, sg):
        """
        Kelime vektör modeli oluşturan metot.
        """
        return Word2Vec(self._corpus, sg=sg, vector_size=100, window=5, min_count=5)

    def __get_corpus(self):
        """
        Model için gerekli kelime sözlüğünü oluşturan metot.
        """
        return [sentence.split() for sentence in self._data if sentence is not None and isinstance(sentence, str)]
