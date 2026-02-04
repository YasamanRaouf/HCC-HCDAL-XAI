# data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path="hcc_dataset.csv"):
    df = pd.read_csv(path)

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    # missing values
    X = pd.DataFrame(X).fillna(np.nanmean(X, axis=0)).values

    # Min-Max normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test
