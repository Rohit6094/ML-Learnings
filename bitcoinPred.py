import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os

def btcDF():
    fpath = 'D:\\codes\\files\\ML-Learnings\\data\\datasets\\novandraanugrah\\bitcoin-historical-datasets-2018-2024\\versions\\346\\btc_1d_data_2018_to_2025.csv'
    data = pd.read_csv(fpath)
    df = pd.DataFrame(data)
    return df

df = btcDF()

# print(df.head())

class EDA:

    def __init__(self,  df):
        self.df = df.copy()
        print(f"Loaded {len(self.df)} daily data from 2018 to 2025 dataset")

    def clean_data(self):
        if ('Ignore') in self.df.columns:
            self.df.drop(columns='Ignore', inplace=True)
        if ('Close time') in self.df.columns:
            self.df.drop(columns='Close time', inplace=True)

        self.df['Open time'] = pd.to_datetime(self.df['Open time'])
        self.df.set_index('Open time', inplace=True)

        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        self.df[cols] = self.df[cols].apply(pd.to_numeric, errors='coerce')

        print(self.df.head())
        print(self.df.info())



btc_eda = EDA(df)
btc_eda.clean_data()
