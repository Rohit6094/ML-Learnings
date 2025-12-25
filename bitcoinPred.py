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

print(df.head())

