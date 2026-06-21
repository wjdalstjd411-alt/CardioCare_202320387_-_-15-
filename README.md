# CardioCare

이번 프로젝트는 심장병 발병 가능성을 예측하는 머신러닝 프로젝트입니다.

그리고 UCI Heart Disease Dataset(저는 통합 데이터셋을 사용하였습니다.)을 활용하여 심장병 위험 여부를 예측하며, MLOps 요소(MLflow, Unit Test, Docker, CI/CD, Monitoring, Drift Detection)를 포함합니다.

---

# 1. 개발 환경

* Python 3.11
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* mlflow
* scipy
* unittest
* logging
* Docker
* GitHub Actions

---

# 2. 프로젝트 구조

```text
.
├── data/
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
├── README.md
└── report.pdf
└── load_data.py #데이터 적재 후 확인용으로 추가했었습니다.
```

---

# 3. 설치 방법

저장소를 복제합니다.

```bash
# 저장소 복제

git clone https://github.com/wjdalstjd411-alt/CardioCare_202320387_-_-15-.git

# 프로젝트 폴더 이동

cd CardioCare_202320387_-_-15-

```

필요 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

---

# 4. 모델 학습

다음 명령어를 실행하여 모델을 학습합니다.

```bash
python src/train.py
```

학습 완료 후 다음 파일이 생성됩니다.

```text
model.pkl
```

또한 MLflow 실험 결과가 기록됩니다.

---

# 5. 모델 추론

```bash
python src/inference.py sample_input.json
```

출력 예시

```json
{
  "predictions": [1, 1],
  "probabilities": [
    [0.45, 0.55],
    [0.17, 0.82]
  ]
}
```

---

# 6. 단위 테스트

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

검증 항목

* 예측 결과 shape 검증
* 예측 확률 범위 검증
* 임상 입력 범위 검증
* 결정론적 예측 검증

---

# 7. Docker 실행

이미지 생성

```bash
docker build -t cardiocare:1.0 .
```

컨테이너 실행

```bash
docker run cardiocare:1.0 sample_input.json
```

---

# 8. 실험 추적 (MLflow)

MLflow를 사용하여 실험을 기록합니다.

기록 항목

* 모델 종류
* Recall
* Precision
* F1 Score
* Balanced Accuracy
* 하이퍼파라미터

---

# 9. 모니터링 및 드리프트 탐지

실행

```bash
python src/monitor.py
```

기능

* logging 기반 추론 로그 생성
* 모델 버전 기록
* 입력 shape 기록
* 예측 결과 기록
* 실제 정답 기록
* KS 검정을 이용한 드리프트 탐지

사용 라이브러리

```python
from scipy.stats import ks_2samp
```

드리프트가 탐지되면 p-value < 0.05 기준으로 플래그를 출력합니다.

---

# 10. CI/CD

GitHub Actions를 사용하여 Push 시 자동으로 수행합니다.

1. Python 환경 구성
2. 의존성 설치
3. 모델 학습
4. Unit Test 실행


CI 통과 여부는 GitHub Actions 탭에서 확인할 수 있습니다.

---

# 11. 최종 모델

최종 모델: Logistic Regression

선정 이유:

심장병 예측 문제에서는 실제 환자를 놓치지 않는 것이 중요하므로 Recall(재현율)을 주요 평가 지표로 사용하였다. 비교한 모델 중 Logistic Regression이 가장 적절한 Recall 성능을 보여 최종 모델로 선정하였다.
