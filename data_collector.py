"""
Module to collect data from Yahoo Finance API
"""

import os
import pandas as pd
import yfinance as yf
import talib

def name(ticker):
    """
    Get the name of the stock from the ticker
    :param ticker: str: Ticker of the stock
    :return: str: Name of the stock
    """

    return ticker.split(".")[0]

def get_data(ticker, start_date, end_date, data_folder):
    """
    Get historical data from Yahoo Finance API
    :param ticker: str: Ticker of the stock
    :param start_date: str: Start date of the historical data
    :param end_date: str: End date of the historical data
    :param data_folder: str: Folder to save the data
    :return: pd.DataFrame: Historical data
    """

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    file_name = f"{data_folder}/{name(ticker)}_{start_date}_{end_date}.feather"

    print(f"Retrieving data: {ticker} - {start_date} to {end_date}")
    if os.path.exists(file_name):
        data = pd.read_feather(file_name)
    else:
        data = download_data(ticker, start_date, end_date, data_folder)

    return data

def download_data(ticker, start_date, end_date, data_folder):
    """
    Download historical data from Yahoo Finance API and save it to a feather file
    :param ticker: str: Ticker of the stock
    :param start_date: str: Start date of the historical data
    :param end_date: str: End date of the historical data
    :param data_folder: str: Folder to save the data
    """

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.shape[0] == 0:
        raise ValueError(
            "No data available for the given date range."
            f"Ticker {ticker} from {start_date} to {end_date}."
        )

    data.drop(["Adj Close"], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.rename(columns={
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Open": "open",
                    "Volume": "volume"
                }, inplace=True)

    # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
    data["feature_close"] = data["close"].pct_change()

    # Create the feature : open[t] / close[t]
    data["feature_open"] = data["open"]/data["close"]

    # Create the feature : high[t] / close[t]
    data["feature_high"] = data["high"]/data["close"]

    # Create the feature : low[t] / close[t]
    data["feature_low"] = data["low"]/data["close"]

    # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
    data["feature_volume"] = data["volume"] / data["volume"].rolling(7*24).max()

    # Create the feature : macd
    data["feature_macd"], data["feature_macdsignal"], data["feature_macdhist"] = talib.MACDFIX(data["close"], signalperiod=9)

    # Create the feature : Three Inside Up/Down
    data["feature_tiud"] = talib.CDL3INSIDE(data["open"], data["high"], data["low"], data["close"])

    # Create the feature : Beta
    data["feature_beta"] = talib.BETA(data["high"], data["low"], timeperiod=5)

    data.dropna(inplace=True) # Clean again !

    data.to_feather(f"{data_folder}/{name(ticker)}_{start_date}_{end_date}.feather")

    return data
