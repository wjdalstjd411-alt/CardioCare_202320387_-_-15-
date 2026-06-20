"""
inference.py — 추론 엔트리포인트
사용법: python src/inference.py <input_json_path>
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

def load_artifact(model_path: str = None) -> dict:
    if model_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(input_data: list, artifact: dict) -> dict:
    df = pd.DataFrame(input_data)

    expected_cols = artifact["feature_columns"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_cols]

    X_processed = artifact["preprocess_pipe"].transform(df)
    X_selected = artifact["selector"].transform(X_processed)

    predictions = artifact["model"].predict(X_selected).tolist()
    probabilities = artifact["model"].predict_proba(X_selected).tolist()

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <input_json_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path, "r") as f:
        input_data = json.load(f)

    artifact = load_artifact()
    result = predict(input_data, artifact)
    print(json.dumps(result, indent=2))