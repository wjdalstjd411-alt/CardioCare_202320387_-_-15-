import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# 1) Data Loading
# =========================
def load_data_from_ucimerged(data_dir: Optional[str] = None) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "..", "data")

    files = [
        os.path.join(data_dir, "processed.cleveland.data"),
        os.path.join(data_dir, "processed.hungarian.data"),
        os.path.join(data_dir, "processed.switzerland.data"),
        os.path.join(data_dir, "processed.va.data"),
    ]

    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "num",
    ]

    df = pd.concat(
        [pd.read_csv(f, header=None, names=columns, na_values="?") for f in files],
        ignore_index=True,
    )
    df["target"] = (df["num"] > 0).astype(int)
    return df

# =========================
# 2) Custom Transformers
# =========================
class DropEmptyOrHighMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, missing_rate_threshold: float = 0.40):
        self.missing_rate_threshold = missing_rate_threshold
        self.columns_to_drop_: Optional[List[str]] = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DropEmptyOrHighMissingColumns expects a pandas DataFrame.")
        missing_rate = X.isna().mean()
        all_nan_cols = missing_rate[missing_rate == 1.0].index.tolist()
        high_missing_cols = missing_rate[missing_rate >= self.missing_rate_threshold].index.tolist()
        self.columns_to_drop_ = sorted(list(set(all_nan_cols + high_missing_cols)))
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns_to_drop_:
            X = X.drop(columns=self.columns_to_drop_, errors="ignore")
        return X

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], k: float = 1.5):
        self.columns = columns
        self.k = k
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IQRClipper expects a pandas DataFrame.")
        self.bounds_ = {}
        for col in self.columns:
            if col not in X.columns:
                continue
            series = X[col].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.k * iqr
            upper = q3 + self.k * iqr
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

# =========================
# 3) Pipeline Builder
# =========================
@dataclass
class PreprocessConfig:
    numeric_cols: List[str]
    categorical_cols: List[str]
    missing_rate_threshold: float = 0.40
    use_knn_imputer: bool = False
    knn_neighbors: int = 5
    iqr_k: float = 1.5
    scale_numeric: bool = True

def build_preprocessing_pipeline(cfg: PreprocessConfig) -> Pipeline:
    if cfg.use_knn_imputer:
        num_imputer = KNNImputer(n_neighbors=cfg.knn_neighbors)
    else:
        num_imputer = SimpleImputer(strategy="median")

    num_steps = [("imputer", num_imputer)]
    if cfg.scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_tf = Pipeline(steps=num_steps)

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre_column = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, cfg.numeric_cols),
            ("cat", categorical_tf, cfg.categorical_cols),
        ],
        remainder="drop",
    )

    full = Pipeline(steps=[
        ("drop_empty_or_high_missing_cols", DropEmptyOrHighMissingColumns(cfg.missing_rate_threshold)),
        ("iqr_clip", IQRClipper(columns=cfg.numeric_cols, k=cfg.iqr_k)),
        ("to_model_matrix", pre_column),
    ])
    return full

# =========================
# 4) Helpers
# =========================
def make_xy(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if "num" in df.columns:
        df = df.drop(columns=["num"])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y

if __name__ == "__main__":
    df = load_data_from_ucimerged()
    X, y = make_xy(df)
    cfg = PreprocessConfig(
        numeric_cols=["age", "trestbps", "chol", "thalach", "oldpeak"],
        categorical_cols=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    )
    pipe = build_preprocessing_pipeline(cfg)
    Xt = pipe.fit_transform(X, y)
    print("X shape:", X.shape)
    print("Transformed shape:", Xt.shape)
    print("y distribution:", y.value_counts().to_dict())