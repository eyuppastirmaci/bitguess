import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from turkishnlp import detector
from snowballstemmer import stemmer
import re
import string


class PreProcess:
    """
    Metin üzerinde ön işleme gerçekleştiren sınıf.
    """

    _nlpDetector = detector.TurkishNLP()
    _nlpDetector.create_word_set()

    def __init__(self, meta_characters, language: str = 'turkish'):
        """
        Yapıcı metot.
        """
        self._meta_characters = meta_characters
        self._stopwords = stopwords.words(language)
        self._porter_stemmer = stemmer(language)
        self._filtered_words = []
        self._rooted_words = []
        self._named_entities = []
        self._lemmatized_words = []
        self._word_net_lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _part_of_speech(sentence: str):
        """
        Cümledeki öğeleri bulan metot.
        """
        return pos_tag(word_tokenize(sentence))

    @staticmethod
    def _clear_urls(sentence: str):
        """
        Metin içindeki linkleri temizleyen metot.
        """
        sentence = re.sub(r"http\S+", "", sentence)
        sentence = re.sub(r"http : //\S+", "", sentence)
        sentence = re.sub(r"https : //\S+", "", sentence)
        sentence = re.sub(r": //\S+", "", sentence)
        return sentence

    @staticmethod
    def _clear_domain(sentence: str):
        """
        Domaini temizleyen metot.
        """
        return re.sub(r"www\S+", "", sentence)

    @staticmethod
    def _clear_punctuation(sentence: str):
        """
        Noktalama işaretlerini temizleyen metot.
        """
        return sentence.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def _clear_number(sentence: str):
        """
        Sayıları Temizleyen metot.
        """
        return re.sub(r'[0-9]+', '', sentence)

    @staticmethod
    def _clear_white_spaces(sentence: str):
        """
        Birden fazla boşluk karakterlerini temizleyen metot.
        """
        return re.sub(' +', ' ', sentence)

    def _stem_words(self, sentence: str):
        """
        Cümlenin köklerini bulan metot.
        """
        if len(self._rooted_words):
            self._rooted_words.clear()
        for word in word_tokenize(sentence):
            self._rooted_words.append(' '.join(self._porter_stemmer.stemWords(word.split())))
        return self._rooted_words

    def _named_entity_recognition(self, sentence: str):
        """
        Cümle içerisinde varlık isim tanıma yapan metot.
        """
        return nltk.ne_chunk(self._part_of_speech(sentence))

    def _clear_meta_characters(self, sentence: str):
        """
        Meta karakterleri temizleyen metot.
        """
        for replace in self._meta_characters:
            sentence = sentence.replace(replace, '')
        return sentence

    def _extract_stop_words(self, sentence: str):
        """
        Cümledeki gereksiz kelimeleri filtreleyen metot.
        """
        if len(self._filtered_words):
            self._filtered_words.clear()
        for word in word_tokenize(sentence):
            if word not in self._stopwords:
                self._filtered_words.append(word)
        return ' '.join(self._filtered_words)

    def fix_typos(self, tweet_list: list):
        """
        Cümle içerisinde ki kelime hatalarını düzeltip yeni liste olarak döndüren metot.
        """
        new_tweet_list = []
        for tweet in tweet_list:
            new_tweet_list.append((' '.join(self._nlpDetector.auto_correct(self._nlpDetector.list_words(tweet)))))
        return new_tweet_list

    def get_stem_words(self, tweet_list: list):
        """
        Cümle içersinde ki kelimeleri kökleriyle güncelleyerek yeni liste olarak döndüren metot.
        """
        new_tweet_list = []
        for tweet in tweet_list:
            new_tweet_list.append((' '.join(self._stem_words(tweet))))
        return new_tweet_list

    def process(self, data_list: list):
        """
        Parametre olarak verilen metinde ön işlene yapan metot.
        """
        new_data_list = []
        for sentence in data_list:
            sentence = str(sentence).lower()
            sentence = self._extract_stop_words(sentence)
            sentence = self._clear_number(sentence)
            sentence = self._clear_meta_characters(sentence)
            sentence = self._clear_urls(sentence)
            sentence = self._clear_domain(sentence)
            new_data_list.append(sentence)
        return new_data_list


class TweetPreProcess(PreProcess):
    """
    Tweet metinleri üzerinde ön işleme gerçekleştiren sınıf.
    """

    def __init__(self, meta_characters):
        """
        Yapıcı metot.
        """
        super().__init__(meta_characters)

    @staticmethod
    def _cleaning_picurl(tweet):
        """
        Resim linkini temizleyen metot.
        """
        if tweet.__contains__("pictwit"):
            tweet = tweet[:tweet.index('pictwit')-1]
        return tweet

    @staticmethod
    def _clear_at(tweet: str):
        """
        @ Karakteriyle başlayan kelimeleri temizleyen metot.
        """
        return re.sub(r"@\S+", "", tweet)

    @staticmethod
    def _clear_hashtags(tweet: str):
        """
        Hashtaglari temizleyen metot.
        """
        return re.sub(r"#\S+", "", tweet)

    def process(self, data_list: list):

        """
        Parametre olarak verilen tweetde ön işlene yapan metot.
        """
        new_data_list = []
        for sentence in data_list:
            sentence = str(sentence).lower()
            sentence = super()._extract_stop_words(sentence)
            sentence = super()._clear_number(sentence)
            sentence = super()._clear_meta_characters(sentence)
            sentence = super()._clear_urls(sentence)
            sentence = super()._clear_domain(sentence)
            sentence = self._clear_at(sentence)
            sentence = self._clear_hashtags(sentence)
            sentence = super()._clear_punctuation(sentence)
            sentence = self._cleaning_picurl(sentence)
            sentence = sentence.strip()
            sentence = super()._clear_white_spaces(sentence)
            new_data_list.append(sentence)
        return new_data_list
