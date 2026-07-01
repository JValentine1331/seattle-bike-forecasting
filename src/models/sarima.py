from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd

def evaluate_sarima(
    train_series,
    test_series,
    order,
    seasonal_order):
    """
    Fit and evaluate a SARIMA model.

    Parameters
    ----------
    train_series : pandas.Series
        Training time series.

    test_series : pandas.Series
        Testing time series.

    order : tuple
        (p, d, q)

    seasonal_order : tuple
        (P, D, Q, m)

    Returns
    -------
    dict
        Dictionary containing model information and
        forecasting performance.
    """

    try:

        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(disp=False)

        forecast = results.get_forecast(
            steps=len(test_series)
        )

        predictions = forecast.predicted_mean

        mae = mean_absolute_error(
            test_series,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                test_series,
                predictions
            )
        )

        mape = mean_absolute_percentage_error(
            test_series,
            predictions
        )

        return {
            "order": order,
            "seasonal_order": seasonal_order,
            "AIC": results.aic,
            "BIC": results.bic,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "success": True
        }

    except Exception as e:

        return {
            "order": order,
            "seasonal_order": seasonal_order,
            "AIC": np.nan,
            "BIC": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "success": False,
            "error": str(e)
        }