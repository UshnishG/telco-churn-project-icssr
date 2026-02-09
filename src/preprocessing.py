import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw telco data
    """
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


    # Drop customerID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables
    """
    df = df.copy()
    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df
