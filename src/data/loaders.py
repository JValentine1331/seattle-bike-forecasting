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