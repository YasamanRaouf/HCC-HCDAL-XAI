import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def load_data(path="Data/hcc-data-complete-balanced.csv", n_features=30):
    df = pd.read_csv(path)

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

    X = df.drop("Class", axis=1).values.astype(float)
    y = df["Class"].values.astype(int)
    feature_names = [c for c in df.columns if c != "Class"]

    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    selector = SelectKBest(mutual_info_classif, k=n_features)
    X = selector.fit_transform(X, y)
    selected_names = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"[Feature Selection] Selected {n_features} features from {len(feature_names)}")
    print(f"Selected features: {selected_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test, selected_names
