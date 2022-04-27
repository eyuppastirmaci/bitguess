import pandas as pd

import pandas_datareader as web

import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
from collections import namedtuple


def fetch_data(path: str, target_currency: str, start: dt.datetime, end: dt.datetime, encoding: str):
    df = web.DataReader(f"BTC-{target_currency}", 'yahoo', start, end)
    df = df.drop(columns=['High', 'Low', 'Open', 'Volume', 'Adj Close'])
    df.to_csv(path, encoding=encoding)


def show_data(path: str, encoding: str, start_date, end_date):
    try:
        df = pd.read_csv(path, encoding=encoding)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] > start_date) & (df['Date'] < end_date)]
        return df['Date'].tolist(), df['Close'].tolist()

    except FileNotFoundError:
        print("Dosya Bulunamadı")

