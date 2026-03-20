import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

CHEAP_THRESHOLD   = 100_000
AVERAGE_THRESHOLD = 350_000

CAT_COLS = ["HallwayType", "HeatingType", "AptManageType",
            "TimeToBusStop", "TimeToSubway", "SubwayStation"]

_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return strip_whitespace(df)

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

def assign_house_labels(df: pd.DataFrame) -> np.ndarray:
    ''''Depending on SalePrice assigns label: cheap (0), average (1) or expensive (2)'''
    return np.select(
        [df["SalePrice"] <= CHEAP_THRESHOLD, df["SalePrice"] <= AVERAGE_THRESHOLD],
        [0, 1], default=2
    )

def prepare_train_data(path: str):
    df = load_data(path)
    df[CAT_COLS] = _enc.fit_transform(df[CAT_COLS])
    y = assign_house_labels(df)
    return df.drop(columns=["SalePrice"]), y

def prepare_test_data(path: str) -> pd.DataFrame:
    df = load_data(path)
    df[CAT_COLS] = _enc.transform(df[CAT_COLS])
    return df
