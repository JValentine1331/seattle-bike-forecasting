"""
Baseline forecasting models.

This module implements simple benchmark forecasting
methods used to evaluate predictive performance before
introducing more sophisticated statistical and machine
learning models.
"""

import pandas as pd


def naive_forecast(series):
    """
    Generate a persistence (naïve) forecast.

    The naïve forecasting model predicts each observation
    using the previous observed value. This serves as a
    simple baseline for evaluating more sophisticated
    forecasting models.

    Parameters
    ----------
    series : pandas.Series
        Historical time series.

    Returns
    -------
    pandas.Series
        Predicted values aligned with the original index.
        The first prediction is NaN because no previous
        observation exists.
    """

    return series.shift(1)

def seasonal_naive_forecast(series, season_length=7):
    """
    Generate a seasonal naïve forecast.

    The seasonal naïve forecasting model predicts each
    observation using the value observed one seasonal
    period earlier. For daily bicycle traffic, a season
    length of seven days captures the weekly commuting
    cycle.

    Parameters
    ----------
    series : pandas.Series
        Historical time series.

    season_length : int, default=7
        Number of observations in one seasonal period.

    Returns
    -------
    pandas.Series
        Predicted values aligned with the original index.
        The first ``season_length`` predictions are NaN
        because no previous seasonal observations exist.
    """

    return series.shift(season_length)

def rolling_mean_forecast(series, window=7):
    
    """
    Generate a rolling mean forecast.

    The rolling mean forecasting model predicts each
    observation using the mean of the previous
    ``window`` observations. Averaging multiple recent
    observations produces a smoother forecast than
    persistence-based methods.

    Parameters
    ----------
    series : pandas.Series
        Historical time series.

    window : int, default=7
        Number of previous observations included in the
        rolling average.

    Returns
    -------
    pandas.Series
        Predicted values aligned with the original index.
        The first ``window`` predictions are NaN because
        insufficient observations exist to compute the
        rolling mean.
    """

    return (series.shift(1).rolling(window=window).mean())
    