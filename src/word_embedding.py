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
        self._path = path
        self._encoding = encoding
        self._corpus = []
        self._model_sg = self.get_model(1)
        self._model_cbow = self.get_model(0)

    @property
    def model_sg(self):
        return self._model_sg

    @property
    def model_cbow(self):
        return self._model_cbow

    def get_model(self, sg):
        """
        Kelime vektör modeli oluşturan metot
        """
        self._corpus.clear()
        for sentence in pd.read_csv(self._path, encoding=self._encoding).text.tolist():
            self._corpus.append(sentence.split())
        return Word2Vec(self._corpus, sg=sg)
