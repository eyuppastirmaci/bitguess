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
        self._corpus = []
        self.data = pd.read_csv(path, encoding=encoding).text.tolist()
        self.word_embedding = self.__get_model(1)
        self.model_cbow = self.__get_model(0)

    def __get_model(self, sg):
        """
        Kelime vektör modeli oluşturan metot
        """
        if len(self._corpus):
            self._corpus.clear()
        for sentence in self.data:
            self._corpus.append(sentence.split())
        return Word2Vec(self._corpus, sg=sg)


