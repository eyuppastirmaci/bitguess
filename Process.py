import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from snowballstemmer import stemmer


class PreProcess:

    filtered_words = []
    rooted_words = []
    named_entities = []
    lemmatized_words = []

    porter_stemmer = stemmer('turkish')
    word_net_lemmatizer = WordNetLemmatizer()

    def __init__(self, language='turkish'):
        """
        Yapıcı metod.
        :param language: Gereksiz kelimelerin hangi dilde ayıklanacağını belirten parametre.
        """
        self.stopwords = stopwords.words(language)

    def extract_stop_words(self, sentence):
        """
        Cümledeki gereksiz kelimeleri filtreleyen metod.
        :param sentence: Gereksiz kelimelerin filtreleneceği cümle.
        :return: Gereksiz kelimelerden filtrelenmiş cümle.
        """
        self.filtered_words.clear()
        for word in word_tokenize(sentence):
            if word not in self.stopwords:
                self.filtered_words.append(word)
        return ' '.join(self.filtered_words)

    def stem_words(self, sentence):
        """
        Cümlenin köklerini bulan metod.
        :param sentence: Kökleri bulunacak cümle.
        :return: Cümledeki köklerin listesi.
        """
        self.rooted_words.clear()
        for word in word_tokenize(sentence):
            self.rooted_words.append(' '.join(self.porter_stemmer.stemWords(word.split())))
        return self.rooted_words

    def part_of_speech(self, sentence):
        """
        Cümledeki öğeleri bulan metod.
        :param sentence: Öğeleri bulunacak cümle.
        :return: Cümlede bulunan öğelerin listesi.
        """
        return pos_tag(word_tokenize(sentence))


    def named_entity_recognition(self, sentence):
        """
        Cümle içerisinde varlık isim tanıma yapan metod.
        :param sentence: Tanımanın yapılacağı cümle.
        :return: Varlık isim tanımlandırılmıi cümle.
        """
        return nltk.ne_chunk(pos_tag(word_tokenize(sentence)))



        













