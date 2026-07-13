# data_loader.py

from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Official column order for the RAW HCC dataset (UCI ML Repository / Kaggle
# "mrsantos/hcc-dataset"), taken from hcc-description.txt distributed with
# the dataset. The raw hcc-data.csv file has NO header row, so the columns
# must be assigned manually in this exact order (49 features + Class).
COLUMN_NAMES = [
    "Gender", "Symptoms", "Alcohol", "HBsAg", "HBeAg", "HBcAb", "HCVAb",
    "Cirrhosis", "Endemic", "Smoking", "Diabetes", "Obesity", "Hemochro",
    "AHT", "CRI", "HIV", "NASH", "Varices", "Spleno", "PHT", "PVT",
    "Metastasis", "Hallmark", "Age", "Grams_day", "Packs_year", "PS",
    "Encephalopathy", "Ascites", "INR", "AFP", "Hemoglobin", "MCV",
    "Leucocytes", "Platelets", "Albumin", "Total_Bil", "ALT", "AST",
    "GGT", "ALP", "TP", "Creatinine", "Nodules", "Major_Dim", "Dir_Bil",
    "Iron", "Sat", "Ferritin", "Class",
]


def load_data(path="Data/hcc-data.csv", n_features=30, random_state=42):
    """
    Loads the RAW HCC dataset: 165 patients, 49 clinical features + Class,
    with a class-imbalanced distribution of 63 "Dies" / 102 "Lives",
    exactly matching the dataset described in the reference paper and in
    hcc-description.txt.

    IMPORTANT -- do NOT load 'hcc-data-complete-balanced.csv' here.
    That file is a pre-imputed, pre-balanced (102/102) artifact produced by
    the cluster-based SMOTE-style oversampling method of Santos et al.
    (2015), applied to the WHOLE dataset before any train/test split.
    Splitting that file directly risks sample-level leakage: a synthetic
    "Dies" patient placed in the test fold can be a near-interpolation of
    real patients placed in the train fold, inflating test performance in a
    way that has nothing to do with the model's real generalization ability.

    The raw file loaded here preserves the paper's original distribution and
    missing values ('?' markers). All preprocessing (imputation, scaling,
    feature selection, and any future class-balancing step) must be fit
    ONLY on the training fold -- which is exactly what the Pipeline below
    does.
    """
    # The raw file has no header and uses '?' as the missing-value marker.
    df = pd.read_csv(path, header=None, na_values="?")

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns in {path}, got {df.shape[1]}. "
            "Make sure you are pointing to the raw 'hcc-data.csv' file "
            "(165 rows, no header), not a pre-processed variant."
        )
    df.columns = COLUMN_NAMES

    X_raw = df.drop("Class", axis=1)
    y = df["Class"].values.astype(int)
    feature_names = list(X_raw.columns)

    # Split BEFORE any preprocessing, so no information about the test set
    # (min/max values, mutual information scores, imputation means, ...)
    # can leak into the training process.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    print(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")
    print(
        f"Train class balance -> Dies: {(y_train == 0).sum()}, "
        f"Lives: {(y_train == 1).sum()}"
    )
    print(
        f"Test  class balance -> Dies: {(y_test == 0).sum()}, "
        f"Lives: {(y_test == 1).sum()}"
    )

    # mutual_info_classif is itself stochastic (it relies on a k-NN based
    # estimator internally). Without seeding it, SelectKBest can pick a
    # DIFFERENT set of 30 features on every run of this script -- which
    # silently changes X_train/X_test and makes downstream accuracy/MCC
    # numbers non-reproducible between runs, even with everything else fixed.
    seeded_mutual_info = partial(mutual_info_classif, random_state=random_state)

    # All preprocessing steps are fit ONLY on the training fold, then simply
    # applied (transform) to the test fold.
    preprocessing_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("selector", SelectKBest(seeded_mutual_info, k=n_features)),
    ])

    X_train = preprocessing_pipeline.fit_transform(X_train_raw, y_train)
    X_test = preprocessing_pipeline.transform(X_test_raw)

    selector = preprocessing_pipeline.named_steps["selector"]
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"Selected {n_features}/{len(feature_names)} features:")
    print(selected_features)

    return X_train, X_test, y_train, y_test, selected_features, preprocessing_pipeline
