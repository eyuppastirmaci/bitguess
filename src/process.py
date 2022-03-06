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

    filtered_words = []
    rooted_words = []
    named_entities = []
    lemmatized_words = []

    porter_stemmer = stemmer('turkish')
    word_net_lemmatizer = WordNetLemmatizer()

    def __init__(self, language: str = 'turkish'):
        """
        Yapıcı metot.
        :param language: Gereksiz kelimelerin hangi dilde ayıklanacağını belirten parametre.
        """
        self.stopwords = stopwords.words(language)

    def extract_stop_words(self, sentence: str):
        """
        Cümledeki gereksiz kelimeleri filtreleyen metot.
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
        Cümlenin köklerini bulan metot.
        :param sentence: Kökleri bulunacak cümle.
        :return: Cümledeki köklerin listesi.
        """
        self.rooted_words.clear()
        for word in word_tokenize(sentence):
            self.rooted_words.append(' '.join(self.porter_stemmer.stemWords(word.split())))
        return self.rooted_words

    @classmethod
    def part_of_speech(cls, sentence: str):
        """
        Cümledeki öğeleri bulan metot.
        :param sentence: Öğeleri bulunacak cümle.
        :return: Cümlede bulunan öğelerin listesi.
        """
        return pos_tag(word_tokenize(sentence))

    def named_entity_recognition(self, sentence: str):
        """
        Cümle içerisinde varlık isim tanıma yapan metot.
        :param sentence: Tanımanın yapılacağı cümle.
        :return: str.
        """
        return nltk.ne_chunk(self.part_of_speech(sentence))

    def process(self, sentence: str):
        """
        Parametre olarak verilen metinde ön işlene yapan metot.
        :param sentence: str
        :return: str
        """
        return self.extract_stop_words(sentence)


class TweetPreProcess(PreProcess):
    """
    Tweet metinleri üzerinde ön işleme gerçekleştiren sınıf.
    """

    def __init__(self, meta_characters):
        super().__init__()
        self.meta_characters = meta_characters

    def clear_meta_characters(self, tweet: str):
        """
        Meta karakterleri temizleyen metot.
        :param tweet: str
        :return: str
        """
        tweet = str(tweet).lower()
        for replace in self.meta_characters:
            tweet = tweet.replace(replace, '')
        return tweet

    @classmethod
    def clear_urls(cls, tweet: str):
        """
        Tweet içindeki linkleri temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r"http\S+", "", re.sub(r"@\S+", "", tweet.lower())).strip()

    @classmethod
    def clear_hashtags(cls, tweet: str):
        """
        Hashtaglari temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r"#\S+", "", tweet).strip()

    @classmethod
    def clear_punctuation(cls, tweet: str):
        """
        Noktalama işaretlerini temizleyen metot.
        :param tweet: str
        :return: str
        """
        return tweet.translate(str.maketrans('', '', string.punctuation))

    @classmethod
    def clear_numbers(cls, tweet: str):
        """
        Sayıları Temizleyen metot.
        :param tweet: str
        :return: str
        """
        return re.sub(r'[0-9]+', '', tweet)

    def process(self, sentence: str):
        """
        Parametre olarak verilen tweetde ön işlene yapan metot.
        :param sentence: str
        :return: str
        """
        sentence = self.extract_stop_words(sentence)
        sentence = self.clear_meta_characters(sentence)
        sentence = self.clear_urls(sentence)
        sentence = self.clear_hashtags(sentence)
        sentence = self.clear_punctuation(sentence)
        sentence = self.clear_numbers(sentence)
        return sentence
