import pandas as pd


def parse_datetime(df, column):
    """
    Convert a DataFrame column to pandas datetime format.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the datetime column.
    column : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with the specified
        column converted to datetime64 format.
    """
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    return df

def interpolate_missing(df, columns):
    """
    Interpolate missing values in selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list[str]
        Columns to interpolate.

    Returns
    -------
    pandas.DataFrame
        DataFrame with interpolated values.
    """
    df = df.copy()

    for col in columns:
        df[col] = df[col].interpolate(method="linear")

    return df

def aggregate_daily(df):
    """
    Aggregate hourly bike and weather data to daily values.

    Parameters
    ----------
    df : pandas.DataFrame
        Hourly bike-weather dataset.

    Returns
    -------
    pandas.DataFrame
        Daily aggregated dataset.
    """
    daily_df = (
        df
        .set_index("date")
        .resample("D")
        .agg(
            {
                "fremont_bridge": "sum",
                "temperature_2m": "mean",
                "precipitation": "sum",
                "wind_speed_10m": "mean",
            }
        )
        .reset_index()
    )

    return daily_df