import pandas as pd

import pandas_datareader as web

import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt


def fetch_data(path: str, target_currency: str, start: dt.datetime, end: dt.datetime, encoding: str):
    df = web.DataReader(f"BTC-{target_currency}", 'yahoo', start, end)
    df = df.drop(columns=['High', 'Low', 'Open', 'Volume', 'Adj Close'])
    df.to_csv(path, encoding=encoding)


def show_data(path: str, encoding: str):
    try:
        df = pd.read_csv(path, encoding=encoding)

        df['Date'] = pd.to_datetime(df['Date'])

        date_list = df['Date'].tolist()
        price_list = df['Close'].tolist()

        plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=25)

        axis = plt.gca()
        axis_date_format = md.DateFormatter('%Y-%m-%d')
        axis.xaxis.set_major_formatter(axis_date_format)

        plt.plot(date_list, price_list)
        plt.show()

    except FileNotFoundError:
        print("Dosya Bulunamadı")

