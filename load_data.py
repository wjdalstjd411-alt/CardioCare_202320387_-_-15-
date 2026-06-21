import pandas as pd

files = [
    "data/processed.cleveland.data",
    "data/processed.hungarian.data",
    "data/processed.switzerland.data",
    "data/processed.va.data",
]

# processed 파일은 헤더가 없으므로 컬럼명을 직접 지정
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num"   # num이 타깃
]

df = pd.concat(
    [pd.read_csv(f, header=None, names=columns, na_values="?") for f in files],
    ignore_index=True,
)
df["target"] = (df["num"] > 0).astype(int)
print(df.shape)
print(df.head())