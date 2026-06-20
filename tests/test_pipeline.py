"""
tests/test_pipeline.py — CardioCare 단위 테스트 (4개 메서드)
"""
import os, sys, unittest, pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def make_sample_input(n: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age": rng.randint(30, 80, n).astype(float),
        "sex": rng.choice([0, 1], n).astype(float),
        "cp": rng.choice([1, 2, 3, 4], n).astype(float),
        "trestbps": rng.randint(90, 200, n).astype(float),
        "chol": rng.randint(100, 400, n).astype(float),
        "fbs": rng.choice([0, 1], n).astype(float),
        "restecg": rng.choice([0, 1, 2], n).astype(float),
        "thalach": rng.randint(70, 210, n).astype(float),
        "exang": rng.choice([0, 1], n).astype(float),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.choice([1, 2, 3], n).astype(float),
        "ca": rng.choice([0, 1, 2, 3], n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })

class TestCardioCare(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.artifact = load_artifact()
        cls.sample = make_sample_input(10)

    def _run_pipeline(self, X):
        expected_cols = self.artifact["feature_columns"]
        X = X[expected_cols]
        X_processed = self.artifact["preprocess_pipe"].transform(X)
        X_selected = self.artifact["selector"].transform(X_processed)
        return X_selected

"""
tests/test_pipeline.py — CardioCare 단위 테스트 (4개 메서드)
"""
import os
import sys
import unittest
import pickle

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def make_sample_input(n=5):
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age": rng.randint(30, 80, n).astype(float),
        "sex": rng.choice([0, 1], n).astype(float),
        "cp": rng.choice([1, 2, 3, 4], n).astype(float),
        "trestbps": rng.randint(90, 200, n).astype(float),
        "chol": rng.randint(100, 400, n).astype(float),
        "fbs": rng.choice([0, 1], n).astype(float),
        "restecg": rng.choice([0, 1, 2], n).astype(float),
        "thalach": rng.randint(70, 210, n).astype(float),
        "exang": rng.choice([0, 1], n).astype(float),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.choice([1, 2, 3], n).astype(float),
        "ca": rng.choice([0, 1, 2, 3], n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })

class TestCardioCare(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.artifact = load_artifact()
        cls.sample = make_sample_input(10)

    def _run_pipeline(self, X):
        expected_cols = self.artifact["feature_columns"]
        X = X[expected_cols]
        X_processed = self.artifact["preprocess_pipe"].transform(X)
        X_selected = self.artifact["selector"].transform(X_processed)
        return X_selected

    # 예측 결과의 shape 가 입력 shape 와 일치하는지
    def test_prediction_shape(self):
        X_selected = self._run_pipeline(self.sample.copy())
        preds = self.artifact["model"].predict(X_selected)
        # preds.shape[0]으로 튜플에서 10을 꺼내어 비교합니다.
        self.assertEqual(preds.shape[0], len(self.sample))


    # 예측 확률이 [0, 1] 범위 내에 있고 행마다 합이 약 1 인지.
    def test_probability_range_and_sum(self):
        X_selected = self._run_pipeline(self.sample.copy())
        probas = self.artifact["model"].predict_proba(X_selected)
        self.assertTrue(np.all(probas >= 0) and np.all(probas <= 1))
        np.testing.assert_allclose(
            probas.sum(axis=1), np.ones(len(probas)), atol=1e-6
        )

    # 임상적으로 범위가 정해진 특성(예: chol 이 [0, 600])에 대한 입력값 범위 검증
    def test_clinical_input_range(self):
        clinical_ranges = {
            "chol": (0, 600),
            "age": (0, 120),
            "trestbps": (0, 300),
            "thalach": (0, 250),
        }
        X = self.sample.copy()
        for col, (lo, hi) in clinical_ranges.items():
            values = X[col].dropna()
            self.assertTrue(
                np.all(values >= lo) and np.all(values <= hi),
                f"{col} out of [{lo}, {hi}]"
            )

    # 고정 시드에서 파이프라인이 결정론적인지 (동일 입력 → 동일 출력).
    def test_deterministic_prediction(self):
        X_selected = self._run_pipeline(self.sample.copy())
        preds_1 = self.artifact["model"].predict(X_selected).copy()
        probas_1 = self.artifact["model"].predict_proba(X_selected).copy()

        X_selected = self._run_pipeline(self.sample.copy())
        preds_2 = self.artifact["model"].predict(X_selected).copy()
        probas_2 = self.artifact["model"].predict_proba(X_selected).copy()

        np.testing.assert_array_equal(preds_1, preds_2)
        np.testing.assert_array_almost_equal(probas_1, probas_2, decimal=10)

if __name__ == "__main__":
    unittest.main()