import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path="Data/hcc-data-complete-balanced.csv"):
    df = pd.read_csv(path)  # standard comma separator

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

    X = df.drop("Class", axis=1).values.astype(float)
    y = df["Class"].values.astype(int)

    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test
