import numpy as np
import pandas as pd
import pandas_datareader as web

import datetime as dts


def fetch_data(path: str, target_currency: str, start: dt.datetime, end: dt.datetime, encoding: str):
    df = web.DataReader(f"BTC-{target_currency}", 'yahoo', start, end)
    df = df.drop(columns=['High', 'Low', 'Open', 'Volume', 'Adj Close'])
    close_list = df['Close'].tolist()
    df['change'] = np.array(pd.Series([0.0] + [b - a for a, b in zip(close_list[::1], close_list[1::1])]))
    df['change_percent'] = np.array(
        pd.Series([0.0] + [100 * (b - a) / a for a, b in zip(close_list[::1], close_list[1::1])]))
    df['sentiment'] = np.zeros(len(close_list))
    df['sentiment_average'] = np.zeros(len(close_list))
    df.to_csv(path, encoding=encoding)


def show_data(path: str, encoding: str, start_date, end_date):
    try:
        df = pd.read_csv(path, encoding=encoding)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] > start_date) & (df['Date'] < end_date)]
        return df['Date'].tolist(), df['Close'].tolist()

    except FileNotFoundError:
        print("Dosya Bulunamadı")

