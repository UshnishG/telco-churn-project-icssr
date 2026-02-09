import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data from given path
    """
    df = pd.read_csv(path)
    return df
