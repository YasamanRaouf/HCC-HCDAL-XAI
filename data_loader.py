import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path="Data/hcc-data-complete-balanced.csv", n_features=30, random_state=42):
    df = pd.read_csv(path)
    
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
    
    X_raw = df.drop("Class", axis=1)
    y = df["Class"].values.astype(int)
    feature_names = list(X_raw.columns)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )
    
    print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")
    
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(mutual_info_classif, k=n_features))
    ])
    
    X_train = preprocessing_pipeline.fit_transform(X_train_raw, y_train)
    X_test = preprocessing_pipeline.transform(X_test_raw)
    
    selector = preprocessing_pipeline.named_steps['selector']
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected {n_features}/{len(feature_names)} features")
    
    return X_train, X_test, y_train, y_test, selected_features, preprocessing_pipeline
