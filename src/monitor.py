"""
04_monitor.py
Monitoring & Drift Detection
"""

import os
import pickle
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot as plt


# =====================================================
# Load Artifact
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
preprocess_pipe = artifact["preprocess_pipe"]
selector = artifact["selector"]

X_test = artifact["X_test"].copy()
y_test = artifact["y_test"].copy()

MODEL_VERSION = "1.0"


# =====================================================
# Logging Setup
# =====================================================

LOG_PATH = os.path.join(PROJECT_ROOT, "monitor.log")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

print("=" * 60)
print("Monitoring Started")
print("=" * 60)


# =====================================================
# Original Prediction
# =====================================================

X_processed = preprocess_pipe.transform(X_test)
X_selected = selector.transform(X_processed)

preds = model.predict(X_selected)

logging.info(
    f"MODEL_VERSION={MODEL_VERSION}"
)

logging.info(
    f"INPUT_SHAPE={X_test.shape}"
)

logging.info(
    f"PREDICTIONS={preds.tolist()}"
)

logging.info(
    f"ACTUAL_LABELS={y_test.tolist()}"
)

orig_bal_acc = balanced_accuracy_score(y_test, preds)

print(f"Original Balanced Accuracy : {orig_bal_acc:.4f}")


# =====================================================
# Artificial Drift
# chol 평균 +30
# 분산 증가
# =====================================================

X_drift = X_test.copy()

if "chol" in X_drift.columns:

    np.random.seed(42)

    X_drift["chol"] = (
        X_drift["chol"].fillna(
            X_drift["chol"].median()
        )
        + 80
        + np.random.normal(0, 40, len(X_drift))
    )


# =====================================================
# KS Test
# =====================================================

print("\nKS Drift Detection")

continuous_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

drift_flags = []

for col in continuous_features:

    base = X_test[col].dropna()
    drift = X_drift[col].dropna()

    stat, p_value = ks_2samp(
        base,
        drift
    )

    flag = p_value < 0.05

    if flag:
        drift_flags.append(col)

    print(
        f"{col:10s} "
        f"p-value={p_value:.6f} "
        f"{'DRIFT' if flag else 'OK'}"
    )

    logging.info(
        f"KS_TEST feature={col} "
        f"p_value={p_value:.6f} "
        f"drift={flag}"
    )


# =====================================================
# Drifted Performance
# =====================================================

X_processed_drift = preprocess_pipe.transform(X_drift)
X_selected_drift = selector.transform(X_processed_drift)

preds_drift = model.predict(X_selected_drift)

drift_bal_acc = balanced_accuracy_score(
    y_test,
    preds_drift
)

print()
print(f"Drifted Balanced Accuracy : {drift_bal_acc:.4f}")
print(
    f"Performance Drop : "
    f"{orig_bal_acc - drift_bal_acc:.4f}"
)

logging.info(
    f"ORIGINAL_BAL_ACC={orig_bal_acc:.4f}"
)

logging.info(
    f"DRIFT_BAL_ACC={drift_bal_acc:.4f}"
)


# =====================================================
# Time Series Visualization
# =====================================================

dates = [
    datetime.now() - timedelta(days=i)
    for i in range(10)
]

dates.reverse()

metric_history = np.linspace(
    orig_bal_acc,
    drift_bal_acc,
    10
)

plt.figure(figsize=(8, 4))

plt.plot(
    dates,
    metric_history,
    marker="o"
)

plt.title(
    "Balanced Accuracy Over Time"
)

plt.ylabel(
    "Balanced Accuracy"
)

plt.xticks(rotation=45)

plt.tight_layout()

FIG_PATH = os.path.join(
    PROJECT_ROOT,
    "monitoring_metric.png"
)

plt.savefig(FIG_PATH)

print(
    f"\nGraph saved : {FIG_PATH}"
)

print(
    f"Log saved   : {LOG_PATH}"
)

print(
    f"Drift Features : {drift_flags}"
)