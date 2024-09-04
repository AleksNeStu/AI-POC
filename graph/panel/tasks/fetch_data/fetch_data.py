import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn

PRIMARY_COLOR = "#0072B5"
SECONDARY_COLOR = "#B54300"
CSV_FILE = (
    "https://raw.githubusercontent.com/holoviz/panel/main/examples/assets/occupancy.csv"
)


pn.extension(design="material", sizing_mode="stretch_width")


@pn.cache
def get_data():
    return pd.read_csv(CSV_FILE, parse_dates=["date"], index_col="date")


data = get_data()

data_part = data.tail()

def transform_data(variable, window, sigma):
    """Calculates the rolling average and identifies outliers"""
    avg = data[variable].rolling(window=window).mean()
    residual = data[variable] - avg
    std = residual.rolling(window=window).std()
    outliers = np.abs(residual) > std * sigma
    return avg, avg[outliers]


def get_plot(variable="Temperature", window=30, sigma=10):
    """Plots the rolling average and the outliers"""
    avg, highlight = transform_data(variable, window, sigma)
    return avg.hvplot(
        height=300, legend=False, color=PRIMARY_COLOR
    ) * highlight.hvplot.scatter(color=SECONDARY_COLOR, padding=0.1, legend=False)

pl = get_plot(variable='Temperature', window=20, sigma=10)


pn.panel(pl).show()