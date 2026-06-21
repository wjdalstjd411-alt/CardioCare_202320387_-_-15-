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
    
    
    #짧은 서술
    #if) 만약 실제 배포를 한다면
    
    #1.피처스토어에 들어갈 한가지 특성은 무엇인가..
     # 피처 스토어에는 혈중 콜레스테롤 수치(chol)를 저장하는 것이 적절하다고 판단하였습니다.
     # 콜레스테롤 수치는 심혈관 질환 위험과 직접적으로 관련된 대표적인 임상 지표이며, 
     # 모델 학습 과정에서도 중요한 예측 변수로 활용되었습니다. 
     # 또한 병원 시스템에서 지속적으로 수집 가능한 데이터이므로 학습 시점과 추론 시점에 동일한 전처리 규칙을 적용하기 쉽습니다.
     # 그리고 이러한 특성 덕분에 시간이 지나면서 성능이 떨어지는 모델 드리프트에도 적절히 대응할 수 있을 것이라고 판단하였습니다.
     # 따라서 chol은 피처 스토어에서 관리해야 할 핵심 피처라고 판단하였습니다.
     
    #2.모델 레지스트리에 기록해야 될 한가지 메타데이터는 무엇인가..
     #모델 레지스트리에는 최종 모델의 Recall 값을 메타데이터로 기록해야 합니다.
     # 저는 이번 프로젝트에서 심장병 환자를 정상으로 잘못 분류하는 False Negative(오탐지)를 최소화하는
     # 것을 목적으로 하였기에 Recall을 주요 성능 지표로 사용했었습니다.
     # 위 피처스토어에 들어갈 chol(콜레스트롤)과 같은 피처에 대한 새로운 데이터가 추가되는 새로운 모델 버전이 등록한다면
     # Recall을 비교하면 환자 탐지 성능이 유지되거나 향상되었는지 쉽게 확인할 수 있을 것입니다.
     # 따라서 Recall은 모델 배포 의사결정에 중요한 메타데이터라고 판단하였습니다.