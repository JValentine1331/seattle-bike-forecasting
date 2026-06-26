"""
Utility functions for evaluating forecasting models.

This module implements common regression and forecasting
error metrics used to compare predicted and observed
bicycle traffic. These metrics provide standardized
measures of forecasting accuracy and are reused
throughout the modeling pipeline.
"""

import numpy as np

def mean_absolute_error(y_true, y_pred):
    
    """
    Compute the Mean Absolute Error (MAE) between observed
    and predicted values.

    Mean Absolute Error is the average absolute difference
    between the observed values and their corresponding
    predictions. Lower values indicate better forecasting
    performance.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean Absolute Error.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    
    """
    Compute the Root Mean Squared Error (RMSE) between
    observed and predicted values.

    Root Mean Squared Error is the square root of the
    average squared prediction error. Because the errors
    are squared before averaging, RMSE penalizes large
    prediction errors more heavily than Mean Absolute
    Error.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Root Mean Squared Error.
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE)
    between observed and predicted values.

    Mean Absolute Percentage Error expresses the average
    prediction error as a percentage of the observed
    values, making it useful for comparing forecasting
    accuracy across datasets with different scales.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean Absolute Percentage Error expressed as a
        percentage.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100