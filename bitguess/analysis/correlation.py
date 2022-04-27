import pandas as pd
from scipy.stats import kendalltau


def correlation():
    df = pd.read_csv('../data/correlation_data.csv', encoding="utf-8", low_memory=False)

    variable_price = df['Close']
    variable_sentiment_average = df['sentiment_average']

    # Kendall’s Tau-b Correlation
    kendalltau_correlation, kendalltau_pvalue = kendalltau(variable_price, variable_sentiment_average)
