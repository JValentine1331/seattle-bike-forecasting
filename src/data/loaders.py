import pandas as pd


def load_csv(filepath):
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(filepath)

def load_bike_counts(filepath):
    """
    Load Seattle bike counter data.
    """
    return pd.read_csv(filepath)

def load_weather(filepath):
    """
    Load weather data from a CSV file.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the weather CSV file.

    Returns
    -------
    pandas.DataFrame
        Weather observations loaded into a DataFrame.
    """
    return pd.read_csv(filepath)

def get_weather_data(start_date, end_date):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=47.6510"
        "&longitude=-122.3470"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
    )

    weather_raw = pd.read_json(url)
    
    weather_df = pd.DataFrame(
        {
            "time": weather_raw.loc["time", "hourly"],
            "temperature_2m": weather_raw.loc["temperature_2m", "hourly"],
            "precipitation": weather_raw.loc["precipitation", "hourly"],
            "wind_speed_10m": weather_raw.loc["wind_speed_10m", "hourly"],
        }
    )

    weather_df["time"] = pd.to_datetime(weather_df["time"])

    return weather_df