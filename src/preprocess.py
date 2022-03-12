import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from snowballstemmer import stemmer
import re
import string


class PreProcess:
    """
    Metin üzerinde ön işleme gerçekleştiren sınıf.
    """

    def __init__(self, language: str = 'turkish'):
        """
        Yapıcı metot.
        :param language: Gereksiz kelimelerin hangi dilde ayıklanacağını belirten parametre.
        """
        self._stopwords = stopwords.words(language)
        self._porter_stemmer = stemmer(language)
        self._word_net_lemmatizer = WordNetLemmatizer()
        self._filtered_words = []
        self._rooted_words = []
        self._named_entities = []
        self._lemmatized_words = []

    @staticmethod
    def __part_of_speech(sentence: str):
        """
        Cümledeki öğeleri bulan metot.
        :param sentence: Öğeleri bulunacak cümle.
        :return: Cümlede bulunan öğelerin listesi.
        """
        return pos_tag(word_tokenize(sentence))

    def __stem_words(self, sentence: str):
        """
        Cümlenin köklerini bulan metot.
        :param sentence: Kökleri bulunacak cümle.
        :return: Cümledeki köklerin listesi.
        """
        if (len(self._rooted_words)):
            self._rooted_words.clear()
        for word in word_tokenize(sentence):
            self._rooted_words.append(' '.join(self._porter_stemmer.stemWords(word.split())))
        return self._rooted_words

    def __named_entity_recognition(self, sentence: str):
        """
        Cümle içerisinde varlık isim tanıma yapan metot.
        :param sentence: Tanımanın yapılacağı cümle.
        :return: str.
        """
        return nltk.ne_chunk(self.__part_of_speech(sentence))

    def _extract_stop_words(self, sentence: str):
        """
        Cümledeki gereksiz kelimeleri filtreleyen metot.
        :param sentence: Gereksiz kelimelerin filtreleneceği cümle.
        :return: Gereksiz kelimelerden filtrelenmiş cümle.
        """
        if (len(self._filtered_words)):
            self._filtered_words.clear()
        for word in word_tokenize(sentence):
            if word not in self._stopwords:
                self._filtered_words.append(word)
        return ' '.join(self._filtered_words)

    def process(self, sentence: str):
        """
        Parametre olarak verilen metinde ön işlene yapan metot.
        :param sentence: str
        :return: str
        """
        return self._extract_stop_words(sentence)


class TweetPreProcess(PreProcess):
    """
    Tweet metinleri üzerinde ön işleme gerçekleştiren sınıf.
    """

    def __init__(self, meta_characters):
        """
        Yapıcı metot.
        """
        super().__init__()
        self.meta_characters = meta_characters

    @staticmethod
    def __cleaning_picurl(tweet):
        """
        Resim linkini temizleyen metot.
        :param tweet: str
        :return: str
        """
        tweet = re.sub(r'pic.twitter.com/[\w]*', "", tweet)
        return tweet

    @staticmethod
    def __clear_urls(tweet: str):
        """
        Tweet içindeki linkleri temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r"http\S+", "", tweet).strip()

    @staticmethod
    def __clear_domain(tweet: str):
        """
        Domaini temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r"www\S+", "", tweet).strip()

    @staticmethod
    def __clear_at(tweet: str):
        """
        @ Karakteriyle başlayan kelimeleri temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r"@\S+", "", tweet).strip()

    @staticmethod
    def __clear_hashtags(tweet: str):
        """
        Hashtaglari temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r"#\S+", "", tweet).strip()

    @staticmethod
    def __clear_punctuation(tweet: str):
        """
        Noktalama işaretlerini temizleyen metot.
        :param tweet: str
        :return: str
        """
        return tweet.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def __clear_numbers(tweet: str):
        """
        Sayıları Temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r'[0-9]+', '', tweet)

    def __clear_meta_characters(self, tweet: str):
        """
        Meta karakterleri temizleyen metot.
        :param tweet: str
        :return: str
        """
        for replace in self.meta_characters:
            tweet = tweet.replace(replace, '')
        return tweet

    def process(self, sentence: str):
        """
        Parametre olarak verilen tweetde ön işlene yapan metot.
        :param sentence: str
        :return: str
        """
        sentence = str(sentence).lower()
        sentence = self._extract_stop_words(sentence)
        sentence = self.__cleaning_picurl(sentence)
        sentence = self.__clear_urls(sentence)
        sentence = self.__clear_domain(sentence)
        sentence = self.__clear_at(sentence)
        sentence = self.__clear_hashtags(sentence)
        sentence = self.__clear_punctuation(sentence)
        sentence = self.__clear_numbers(sentence)
        sentence = sentence.replace("  ", " ")
        sentence = self.__clear_meta_characters(sentence).strip()
        return sentence
