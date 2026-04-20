
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")

# reproducibility
SEED = 42
TEST_SIZE = 0.2


def load_raw():
    """Read the CSV and return the full DataFrame."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}.\n"
            "Download it from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv inside the data/ folder."
        )
    return pd.read_csv(DATA_FILE)


def preprocess(df):
    df = df.copy()

    scaler = StandardScaler()
    df["Time"]   = scaler.fit_transform(df[["Time"]])
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values
    return X, y, scaler, feature_cols


def split_data(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )

    #keep only normal (class 0) for training
    normal_mask = y_train_full == 0
    X_train = X_train_full[normal_mask]
    y_train = y_train_full[normal_mask]

    print(f"Training set : {X_train.shape[0]:,} transactions (all normal)")
    print(f"Test set     : {X_test.shape[0]:,} transactions "
          f"({(y_test == 1).sum()} fraud, {(y_test == 0).sum()} normal)")
    return X_train, X_test, y_train, y_test


def get_data():
    """Full pipeline: load → preprocess → split. Returns everything."""
    df = load_raw()

    print(f"Loaded {len(df):,} transactions  "
          f"({(df['Class'] == 1).sum()} fraud = "
          f"{(df['Class'] == 1).mean():.3%})")

    X, y, scaler, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, scaler, feature_cols


if __name__ == "__main__":
    get_data()
