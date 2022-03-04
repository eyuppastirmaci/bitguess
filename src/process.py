import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from snowballstemmer import stemmer
import re


class PreProcess:
    """
    Metin üzerinde ön işleme gerçekleştiren sınıf.
    """

    filtered_words = []
    rooted_words = []
    named_entities = []
    lemmatized_words = []


    porter_stemmer = stemmer('turkish')
    word_net_lemmatizer = WordNetLemmatizer()

    def __init__(self, language: str = 'turkish'):
        """
        Yapıcı metod.
        :param language: Gereksiz kelimelerin hangi dilde ayıklanacağını belirten parametre.
        """
        self.stopwords = stopwords.words(language)

    def extract_stop_words(self, sentence: str):
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

    def stem_words(self, sentence: str):
        """
        Cümlenin köklerini bulan metod.
        :param sentence: Kökleri bulunacak cümle.
        :return: Cümledeki köklerin listesi.
        """
        self.rooted_words.clear()
        for word in word_tokenize(sentence):
            self.rooted_words.append(' '.join(self.porter_stemmer.stemWords(word.split())))
        return self.rooted_words

    def part_of_speech(self, sentence: str):
        """
        Cümledeki öğeleri bulan metod.
        :param sentence: Öğeleri bulunacak cümle.
        :return: Cümlede bulunan öğelerin listesi.
        """
        return pos_tag(word_tokenize(sentence))

    def named_entity_recognition(self, sentence: str):
        """
        Cümle içerisinde varlık isim tanıma yapan metod.
        :param sentence: Tanımanın yapılacağı cümle.
        :return: Varlık isim tanımlandırılmıi cümle.
        """
        return nltk.ne_chunk(pos_tag(word_tokenize(sentence)))


class TweetPreProcess(PreProcess):
    """
    Tweet metinleri üzerinde ön işleme gerçekleştiren class.
    """

    meta_characters = ["rt", "\n", "\t"]

    def __init__(self):
        super().__init__()

    def clear_meta_characters(self, tweet: str):
        """
        Meta karakterleri temizleyen metod.
        :param tweet: str
        :return: Meta kararkterlerden temizlenmiş tweet.
        """
        tweet = str(tweet).lower()
        for replace in self.meta_characters:
            tweet = tweet.replace(replace, '')
        return tweet

    def clear_urls(self, tweet: str):
        """
        Tweet içindeki linkleri temizleyen metod.
        :param tweet: str
        :return: Linklerden temizlenmiş tweet.
        """
        return re.sub(r"http\S+", re.sub(r"@\S+", "", tweet.lower())).strip()

