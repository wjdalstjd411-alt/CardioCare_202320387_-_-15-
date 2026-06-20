"""
train.py — CardioCare: 특성공학, 모델 학습, 실험 추적, 모델 저장
"""
import os, sys, warnings, pickle

# ── 환경 설정 (import 전에 위치해야 함) ──
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn

sys.path.append(os.path.dirname(__file__))
from preprocessing import (
    load_data_from_ucimerged, make_xy,
    PreprocessConfig, build_preprocessing_pipeline,
)

warnings.filterwarnings("ignore")

# =====================================================
# 0. Configuration
# =====================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MLFLOW_EXPERIMENT = "CardioCare"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "model.pkl")

# =====================================================
# 1. Data Loading & Split
# =====================================================
print("=" * 60)
print("1. Data Loading & Train-Test Split")
print("=" * 60)

df = load_data_from_ucimerged()
X, y = make_xy(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
)

# 중복 제거 (X와 y를 함께)
train_df = X_train.copy()
train_df["target"] = y_train.values
train_df = train_df.drop_duplicates().reset_index(drop=True)
y_train = train_df["target"].astype(int)
X_train = train_df.drop(columns=["target"])

print(f"  Total samples : {len(X)}")
print(f"  Train samples : {len(X_train)}  ({1-TEST_SIZE:.0%})")
print(f"  Test  samples : {len(X_test)}  ({TEST_SIZE:.0%})")
print(f"  random_state  : {RANDOM_STATE}")
print(f"  stratify      : y (target class ratio preserved)")
print(f"  Train target distribution:\n{y_train.value_counts(normalize=True).round(4)}")
print()

# =====================================================
# 2. Preprocessing Pipeline (fit on train only)
# =====================================================
print("=" * 60)
print("2. Preprocessing (fit on train only)")
print("=" * 60)

cfg = PreprocessConfig(
    numeric_cols=NUMERIC_COLS,
    categorical_cols=CATEGORICAL_COLS,
    missing_rate_threshold=0.40,
    use_knn_imputer=False,
    iqr_k=1.5,
    scale_numeric=True,
)

preprocess_pipe = build_preprocessing_pipeline(cfg)
X_train_processed = preprocess_pipe.fit_transform(X_train, y_train)
X_test_processed = preprocess_pipe.transform(X_test)

print(f"  X_train shape after preprocessing: {X_train_processed.shape}")
print(f"  X_test  shape after preprocessing: {X_test_processed.shape}")
print()

# =====================================================
# 3. Feature Selection
# =====================================================
print("=" * 60)
print("3. Feature Selection (RF-based SelectFromModel)")
print("=" * 60)

fs_rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
fs_rf.fit(X_train_processed, y_train)

selector = SelectFromModel(fs_rf, prefit=True)
X_train_selected = selector.transform(X_train_processed)
X_test_selected = selector.transform(X_test_processed)

selected_mask = selector.get_support()
importances = fs_rf.feature_importances_

try:
    feature_names = preprocess_pipe.named_steps["to_model_matrix"].get_feature_names_out()
except Exception:
    feature_names = [f"feature_{i}" for i in range(X_train_processed.shape)]

selected_features = [f for f, s in zip(feature_names, selected_mask) if s]

print(f"  Total features before selection : {X_train_processed.shape}")
print(f"  Selected features               : {X_train_selected.shape}")
print(f"\n  Selected feature list:")
for fname in selected_features:
    idx = list(feature_names).index(fname)
    print(f"    - {fname}  (importance: {importances[idx]:.4f})")
print()

# =====================================================
# 4. Model Definitions (4 families)
# =====================================================
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "family": "linear",
    },
    "SVC": {
        "model": SVC(probability=True, random_state=RANDOM_STATE),
        "family": "kernel",
    },
    "RandomForest": {
        "model": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        "family": "ensemble_tree",
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "family": "instance_based",
    },
}

# =====================================================
# 5. 5-Fold CV + MLflow Logging
# =====================================================
print("=" * 60)
print("5. 5-Fold CV + MLflow Logging")
print("=" * 60)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

mlflow.set_tracking_uri("file:///" + os.path.join(PROJECT_ROOT, "mlruns").replace("\\", "/"))
mlflow.set_experiment(MLFLOW_EXPERIMENT)

cv_summary = {}

for name, spec in models.items():
    clf = spec["model"]
    family = spec["family"]

    cv_result = cross_validate(
        clf, X_train_selected, y_train,
        cv=cv, scoring=scoring, n_jobs=1,
    )

    cv_metrics = {
        "cv_balanced_accuracy_mean": float(cv_result["test_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(cv_result["test_balanced_accuracy"].std()),
        "cv_precision_mean": float(cv_result["test_precision"].mean()),
        "cv_recall_mean": float(cv_result["test_recall"].mean()),
        "cv_f1_mean": float(cv_result["test_f1"].mean()),
    }

    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)

    test_metrics = {
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
    }

    cm = confusion_matrix(y_test, y_pred)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", family)
        mlflow.set_tag("model_name", name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_features_selected", X_train_selected.shape)
        mlflow.log_param("model_class", type(clf).__name__)

        params = clf.get_params()
        for k, v in params.items():
            try:
                mlflow.log_param(f"model__{k}", v)
            except Exception:
                pass

        for k, v in cv_metrics.items():
            mlflow.log_metric(k, round(v, 4))
        for k, v in test_metrics.items():
            mlflow.log_metric(k, round(v, 4))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Heart Disease"]).plot(ax=ax, cmap="Blues")
        ax.set_title(f"{name} - Confusion Matrix")
        cm_path = os.path.join(PROJECT_ROOT, f"confusion_matrix_{name}.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        mlflow.sklearn.log_model(sk_model=clf, name="model", serialization_format="pickle")

    print(f"\n  [{name}] (family: {family})")
    print(f"    CV  balanced_acc: {cv_metrics['cv_balanced_accuracy_mean']:.4f} +/- {cv_metrics['cv_balanced_accuracy_std']:.4f}")
    print(f"    CV  recall:       {cv_metrics['cv_recall_mean']:.4f}")
    print(f"    CV  F1:           {cv_metrics['cv_f1_mean']:.4f}")
    print(f"    Test balanced_acc: {test_metrics['test_balanced_accuracy']:.4f}")
    print(f"    Test recall:       {test_metrics['test_recall']:.4f}")
    print(f"    Test F1:           {test_metrics['test_f1']:.4f}")
    print(f"    Test precision:    {test_metrics['test_precision']:.4f}")
    print(f"    Confusion Matrix:\n{cm}")

    cv_summary[name] = {**cv_metrics, **test_metrics, "confusion_matrix": cm.tolist()}

# =====================================================
# 6. Hyperparameter Tuning (LogisticRegression)
# =====================================================
print("\n" + "=" * 60)
print("6. Hyperparameter Tuning (GridSearchCV on LogisticRegression)")
print("=" * 60)

TUNING_SCORING = "recall"

param_grid = [
    {"solver": ["lbfgs", "liblinear"], "penalty": ["l2"],
     "C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    {"solver": ["liblinear", "saga"], "penalty": ["l1"],
     "C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
]

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
    param_grid=param_grid,
    cv=cv,
    scoring=TUNING_SCORING,
    n_jobs=1,
    verbose=0,
)

grid_search.fit(X_train_selected, y_train)
best_lr = grid_search.best_estimator_
y_pred_tuned = best_lr.predict(X_test_selected)

tuned_metrics = {
    "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_tuned)),
    "test_precision": float(precision_score(y_test, y_pred_tuned)),
    "test_recall": float(recall_score(y_test, y_pred_tuned)),
    "test_f1": float(f1_score(y_test, y_pred_tuned)),
}
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

with mlflow.start_run(run_name="LogisticRegression_Tuned"):
    mlflow.set_tag("model_family", "linear")
    mlflow.set_tag("model_name", "LogisticRegression_Tuned")
    mlflow.set_tag("tuning", "GridSearchCV")
    mlflow.set_tag("tuning_scoring", TUNING_SCORING)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("n_features_selected", X_train_selected.shape)

    best_params = grid_search.best_params_
    for k, v in best_params.items():
        mlflow.log_param(f"best__{k}", v)

    mlflow.log_metric("best_cv_score", round(float(grid_search.best_score_), 4))
    for k, v in tuned_metrics.items():
        mlflow.log_metric(k, round(v, 4))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_tuned, display_labels=["Normal", "Heart Disease"]).plot(ax=ax, cmap="Blues")
    ax.set_title("LogisticRegression (Tuned) - Confusion Matrix")
    cm_path = os.path.join(PROJECT_ROOT, "confusion_matrix_LR_Tuned.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    mlflow.sklearn.log_model(sk_model=best_lr, name="model", serialization_format="pickle")

print(f"  Best params: {best_params}")
print(f"  Best CV {TUNING_SCORING}: {grid_search.best_score_:.4f}")
print(f"  Test balanced_accuracy:    {tuned_metrics['test_balanced_accuracy']:.4f}")
print(f"  Test recall:               {tuned_metrics['test_recall']:.4f}")
print(f"  Test F1:                   {tuned_metrics['test_f1']:.4f}")
print(f"  Test precision:            {tuned_metrics['test_precision']:.4f}")
print(f"  Confusion Matrix:\n{cm_tuned}")

# =====================================================
# 7. Final Model Selection & Clinical Justification
# =====================================================
print("\n" + "=" * 60)
print("7. Final Model Selection & Clinical Justification")
print("=" * 60)

# 최종 모델: 기본 LogisticRegression (테스트 성능 기준 최우수)
final_model_clf = models["LogisticRegression"]["model"]
final_model_clf.fit(X_train_selected, y_train)

clinical_justification = """
[Final Model: Logistic Regression]

1. 심장질환 예측에서 가장 치명적인 오류는 False Negative(실제 환자를 정상으로
   오분류)이다. Logistic Regression은 테스트 세트 기준 recall이 비교 대상
   모델 중 가장 높아, 실제 환자를 놓칠 위험을 최소화한다.

2. 동시에 balanced accuracy와 F1-score도 전체 모델 중 가장 높아, 정상인을
   환자로 잘못 분류하는 False Positive도 적절히 억제되었다.

3. Logistic Regression은 각 특성의 계수(coefficient)를 통해 예측 근거를
   직관적으로 해석할 수 있어, 임상의가 모델의 판단 이유를 이해하고
   신뢰하기에 적합하다. 이는 "inform, not decide" 원칙에 부합한다.

4. 5-fold 교차검증 balanced accuracy(0.7858)와 테스트 세트 성능(0.8314)
   간 큰 괴리가 없어 과적합 위험이 낮으며, 새로운 환자 데이터에도
   안정적으로 일반화될 것으로 판단하였다.
   
5.  recall값 증가를 목표로 하이퍼파라미터 튜닝을 실행했을 때 검증 데이터와 테스트
    데이터의 balanced accuracy 점수 사이에 큰 괴리감이 있어, 과적합으로 판단했다.
    그래서 gpt에게 해결책을 물어봤을 때 recall값 증가가 아닌 balanced accuracy의
    증가를 추천하여서 다시 train 해보았을 때 과적합은 막았지만 전체적인 지표가 튜닝전
    모델보다 못하기에 기본 LogisticRegression을 최종모델로 선정하였다.
"""
print(clinical_justification)

# =====================================================
# 8. Save Final Model Artifact (model.pkl at project root)
# =====================================================
print("=" * 60)
print("8. Save Final Model & Summary")
print("=" * 60)

artifact = {
    "preprocess_pipe": preprocess_pipe,
    "selector": selector,
    "model": final_model_clf,
    "feature_columns": list(X_train.columns),
    "random_state": RANDOM_STATE,
}

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(artifact, f)

print(f"  Model saved to {MODEL_SAVE_PATH}")

# Summary Table
summary_rows = []
for name, metrics in cv_summary.items():
    summary_rows.append({
        "Model": name,
        "CV_BalAcc": round(metrics["cv_balanced_accuracy_mean"], 4),
        "CV_Recall": round(metrics["cv_recall_mean"], 4),
        "CV_F1": round(metrics["cv_f1_mean"], 4),
        "Test_BalAcc": round(metrics["test_balanced_accuracy"], 4),
        "Test_Recall": round(metrics["test_recall"], 4),
        "Test_F1": round(metrics["test_f1"], 4),
        "Test_Precision": round(metrics["test_precision"], 4),
    })

summary_rows.append({
    "Model": "LR_Tuned",
    "CV_BalAcc": round(grid_search.best_score_, 4),
    "CV_Recall": "-",
    "CV_F1": "-",
    "Test_BalAcc": round(tuned_metrics["test_balanced_accuracy"], 4),
    "Test_Recall": round(tuned_metrics["test_recall"], 4),
    "Test_F1": round(tuned_metrics["test_f1"], 4),
    "Test_Precision": round(tuned_metrics["test_precision"], 4),
})

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print(f"\n  All runs logged to MLflow experiment: {MLFLOW_EXPERIMENT}")
print(f"  Final Selected Model: LogisticRegression")