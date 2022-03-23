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

    def __init__(self, meta_characters, is_stem_words: bool = False, is_typo_fix: bool = False, language: str = 'turkish'):
        """
        Yapıcı metot.
        """
        self.meta_characters = meta_characters
        self.is_stem_words = is_stem_words
        self.is_typo_fix = is_typo_fix
        self._stopwords = stopwords.words(language)
        self._porter_stemmer = stemmer(language)
        self._filtered_words = []
        self._rooted_words = []
        self._named_entities = []
        self._lemmatized_words = []
        self.nlpDetector = detector.TurkishNLP()
        self.nlpDetector.create_word_set()
        self.word_net_lemmatizer = WordNetLemmatizer()

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
        sentence = re.sub(r"http\S+", "", sentence).strip()
        sentence = re.sub(r"http : //\S+", "", sentence).strip()
        sentence = re.sub(r"https : //\S+", "", sentence).strip()
        sentence = re.sub(r": //\S+", "", sentence).strip()
        return sentence

    @staticmethod
    def _clear_domain(sentence: str):
        """
        Domaini temizleyen metot.
        """
        return re.sub(r"www\S+", "", sentence).strip()

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

    def _fix_typos(self, sentence: str):
        """
        Cümle içerisindeki kelime hatalarını düzelten metot.
        """
        return ' '.join(self.nlpDetector.auto_correct(self.nlpDetector.list_words(sentence)))

    def _get_stem_words(self, sentence: str):
        return ' '.join(self._stem_words(sentence))

    def _clear_meta_characters(self, sentence: str):
        """
        Meta karakterleri temizleyen metot.
        """
        for replace in self.meta_characters:
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

    def _normalization(self, sentence):
        """
        Belirli ön işleme metotlarının tek bir fonksiyonda toplanması.
        """
        sentence = str(sentence).lower()
        sentence = self._extract_stop_words(sentence)
        sentence = self._clear_punctuation(sentence)
        sentence = self._clear_number(sentence)
        sentence = self._clear_meta_characters(sentence).strip()
        sentence = self._clear_urls(sentence)
        sentence = self._clear_domain(sentence)
        sentence = self._clear_white_spaces(sentence)
        if self.is_typo_fix:
            sentence = self._fix_typos(sentence)
        if self.is_stem_words:
            sentence = self._get_stem_words(sentence)
        return sentence

    def process(self, sentence: str):
        """
        Parametre olarak verilen metinde ön işlene yapan metot.
        """
        return self._normalization(sentence)


class TweetPreProcess(PreProcess):
    """
    Tweet metinleri üzerinde ön işleme gerçekleştiren sınıf.
    """

    def __init__(self, meta_characters, is_stem_words, is_typo_fix):
        """
        Yapıcı metot.
        """
        super().__init__(meta_characters, is_stem_words, is_typo_fix)

    @staticmethod
    def _cleaning_picurl(tweet):
        """
        Resim linkini temizleyen metot.
        """
        tweet = re.sub(r'pic.twitter.com/[\w]*', "", tweet)
        return tweet

    @staticmethod
    def _clear_at(tweet: str):
        """
        @ Karakteriyle başlayan kelimeleri temizleyen metot.
        """
        return re.sub(r"@\S+", "", tweet).strip()

    @staticmethod
    def _clear_hashtags(tweet: str):
        """
        Hashtaglari temizleyen metot.
        """
        return re.sub(r"#\S+", "", tweet).strip()

    def process(self, sentence: str):

        """
        Parametre olarak verilen tweetde ön işlene yapan metot.
        """
        sentence = self._normalization(sentence)
        sentence = self._cleaning_picurl(sentence)
        sentence = self._clear_at(sentence)
        sentence = self._clear_hashtags(sentence)
        return sentence
