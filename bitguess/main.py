import sys

from PyQt5.QtGui import QFont

from bitguess.analysis.correlation import correlation
from bitguess.process.preprocess import TweetPreProcess
from bitguess.file.data_file import DataFile
from bitguess.process.embedding import WordVector
from analysis.regression import LogisticRegressionModel
from analysis.sentiment import RnnGruModel
from currency import data

import datetime as dt

from PyQt5 import QtCore
from PyQt5.QtWidgets import *

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure




def preprocessing(data_path, encoding, meta_characters, column_index, out_path):
    # Ön işleme
    preprocess = TweetPreProcess(meta_characters=meta_characters)
    data_file = DataFile(data_path, preprocess, encoding)
    data_file.pre_process_column(column_index, out_path)


def word_embedding(encoding, out_path):
    # Kelime gömme.
    word_vector = WordVector(out_path, encoding)
    word_vector.run()


def logistic_regression(out_path):
    # Lojistik Regresyon modeli kullanarak regresyon analizi.

    MAX_VECTORIZER_FEATURES = 100_000
    MAX_MODEL_ITER = 2_000

    logistic_regression_model = LogisticRegressionModel(out_path,
                                                        max_vectorizer_features=MAX_VECTORIZER_FEATURES,
                                                        max_model_iter=MAX_MODEL_ITER)
    logistic_regression_model.run()


def sentiment_analysis():
    # RNN-GRU modeli kullanarak duygu analizi.

    TRAINING_RATIO = 0.7
    ANALYSIS_DATA_PATH = "../data/sentiment_analysis_data.csv"

    rnn_gru_model = RnnGruModel(data_path=ANALYSIS_DATA_PATH,
                                training_ratio=TRAINING_RATIO)
    rnn_gru_model.run()


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class WindowApp(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(WindowApp, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Bit Guess')
        self.resize(1300, 800)
        self.center()

        self.hbox_main = QHBoxLayout()


        self.vbox_left_panel = QVBoxLayout()
        self.vbox_left_panel.setAlignment(QtCore.Qt.AlignTop)
        self.vbox_right_panel = QVBoxLayout()

        self.hbox_main.addLayout(self.vbox_left_panel)
        self.hbox_main.addLayout(self.vbox_right_panel)

        self.hbox_top_bar = QHBoxLayout()
        self.vbox_left_panel.addLayout(self.hbox_top_bar)

        self.plot_box = QVBoxLayout()
        self.vbox_left_panel.addLayout(self.plot_box)

        btn_show_price_graph = QPushButton("Analiz Et")
        btn_show_price_graph.clicked.connect(lambda x: self.analyze())

        self.dateedit_start = QDateEdit(calendarPopup=True)
        self.dateedit_start.setDateTime(QtCore.QDateTime.currentDateTime())

        self.dateedit_end = QDateEdit(calendarPopup=True)
        self.dateedit_end.setDateTime(QtCore.QDateTime.currentDateTime())

        self.hbox_top_bar.addWidget(btn_show_price_graph)
        self.hbox_top_bar.addWidget(self.dateedit_start)
        self.hbox_top_bar.addWidget(self.dateedit_end)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.hbox_main)

        self.v_widget = QWidget()
        self.v_widget.setFixedWidth(170)
        self.vbox_right_panel.addWidget(self.v_widget)

        self.vbox_correlation_types = QVBoxLayout()
        self.vbox_correlation_types.setSpacing(16)
        self.vbox_correlation_types.setAlignment(QtCore.Qt.AlignTop)
        self.v_widget.setLayout(self.vbox_correlation_types)

        lbl_correlation_type_title = QLabel("Korelasyon Tipi")
        lbl_correlation_type_title.setAlignment(QtCore.Qt.AlignCenter)
        lbl_correlation_type_title.setFont(QFont('Open Sans Bold', 16))

        self.radio_button_1 = QRadioButton("Kendall’s Tau-b")
        self.radio_button_1.setFont(QFont('Open Sans Bold', 12))

        self.radio_button_2 = QRadioButton("Point-biserial")
        self.radio_button_2.setFont(QFont('Open Sans Bold', 12))

        self.vbox_correlation_types.addWidget(lbl_correlation_type_title)

        self.vbox_correlation_choice_container = QVBoxLayout()
        self.vbox_correlation_choice_container.setSpacing(8)

        self.vbox_correlation_types.addLayout(self.vbox_correlation_choice_container)

        self.vbox_correlation_choice_container.addWidget(self.radio_button_1, alignment=QtCore.Qt.AlignLeft)
        self.vbox_correlation_choice_container.addWidget(self.radio_button_2, alignment=QtCore.Qt.AlignLeft)

        self.setCentralWidget(self.central_widget)

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def analyze(self):
        # Eski grafiği silmek için.
        for i in reversed(range(self.plot_box.count())):
            self.plot_box.itemAt(i).widget().setParent(None)

        start_date = str(self.dateedit_start.date().toPyDate())
        end_date = str(self.dateedit_end.date().toPyDate())

        graph = MplCanvas(self, width=5, height=10, dpi=100)
        price_list, sentiment_list = data.show_data(path="../data/bitcoin_data.csv",
                                                    encoding='utf-8',
                                                    start_date=start_date,
                                                    end_date=end_date)
        graph.axes.plot(price_list, sentiment_list)
        self.plot_box.addWidget(graph)

        correlation_value = 0
        p_value = 0

        label = QLabel(f"Korelasyon Sonucu: {correlation_value}\tP Değeri: {p_value}")
        label.setFont(QFont('Open Sans Bold', 14))
        self.plot_box.addWidget(label)


def main():
    # Parametreler
    ENCODING = 'utf-8'
    META_CHARACTERS = ["rt ", "\n", "\t"]
    COLUMN_INDEX = 8
    DATA_PATH = "../data/data.csv"
    OUT_PATH = "../data/preprocessed_data.csv"

    BTC_DATA_PATH = "../data/bitcoin_data.csv"
    BTC_TO = "USD"
    FETCH_START_DATE = dt.datetime(2017, 1, 1)
    FETCH_END_DATE = dt.datetime.now()

    # preprocessing(DATA_PATH, ENCODING, META_CHARACTERS, COLUMN_INDEX, OUT_PATH)

    # word_embedding(ENCODING, OUT_PATH)

    # logistic_regression(OUT_PATH)

    # sentiment_analysis()

    """
    data.fetch_data(path=BTC_DATA_PATH,
                    target_currency=BTC_TO,
                    start=FETCH_START_DATE,
                    end=FETCH_END_DATE,
                    encoding=ENCODING)
    """

    # correlation()

    app = QApplication(sys.argv)
    ex = WindowApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
