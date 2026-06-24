import pandas as pd

def add_calendar_features(df):
    """
    Create calendar-based features from a datetime column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a datetime column named
       'date'.
    
    Returns
    -------
    pandas.DataFrame
       Copy of the input DataFrame with the following
       calendar features added:
    
           - day_of_week : int
               Day of the week (Monday=0, Sunday=6).
           - month : int
               Month of the year (1-12).
           - year : int
               Calendar year.
           - day_of_year : int
               Day number within the year (1-366).
           - is_weekend : int
               Indicator variable where 1 represents Saturday
               or Sunday and 0 represents a weekday.
    
    Notes
    -----
    Calendar features capture weekly and annual seasonality
    patterns that commonly influence bicycle traffic demand.
    These features are useful predictors for forecasting
    models and exploratory data analysis.
    """

    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_name"] = (df["date"].dt.month_name())
    
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    month_order = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December"
    ]
    
    df["month_name"] = pd.Categorical(
        df["date"].dt.month_name(),
        categories=month_order,
        ordered=True
    )
   

    return df

def add_lag_features(df):
    """
    Create lagged traffic features from historical observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a target variable column
        named 'fremont_bridge'.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with the following lag
        features added:

        - lag_1 : float
            Bicycle traffic count from the previous day.
        - lag_7 : float
            Bicycle traffic count from seven days prior.

    Notes
    -----
    Lag features capture temporal dependence (autocorrelation)
    in the time series. Bicycle traffic on a given day is often
    related to traffic on previous days, making lag variables
    important predictors for forecasting models such as linear
    regression, random forests, and gradient boosting methods.

    The first observations in each lagged feature will contain
    missing values because insufficient historical data exists
    to compute the lag.
    """
    
    df = df.copy()

    df["lag_1"] = df["fremont_bridge"].shift(1)

    df["lag_7"] = df["fremont_bridge"].shift(7)

    return df

def add_rolling_features(df):
    """
    Create rolling-window summary statistics from historical
    bicycle traffic observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a target variable column
        named 'fremont_bridge'.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with the following rolling
        features added:

        - rolling_mean_7 : float
            Seven-day moving average of bicycle traffic.
        - rolling_mean_30 : float
            Thirty-day moving average of bicycle traffic.

    Notes
    -----
    Rolling features smooth short-term fluctuations and capture
    recent trends in bicycle traffic. These features provide
    information about local behavior that may not be evident
    from individual daily observations.

    Rolling-window statistics are commonly used in forecasting
    and time-series analysis to represent evolving traffic
    patterns and underlying trends.

    The first observations in each rolling feature will contain
    missing values because insufficient historical data exists
    to compute the rolling average.
    """
    df = df.copy()

    df["rolling_mean_7"] = (
        df["fremont_bridge"]
        .rolling(7)
        .mean()
    )

    df["rolling_mean_30"] = (
        df["fremont_bridge"]
        .rolling(30)
        .mean()
    )

    return df