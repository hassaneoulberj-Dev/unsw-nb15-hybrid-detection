"""Predict labels from a trusted, locally trained pipeline."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.training import feature_frame, hidden_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model_dir / "pipeline.joblib")
    original = pd.read_csv(args.input)
    x = feature_frame(original)[bundle["feature_columns"]]
    x = bundle["preprocessor"].transform(x).astype(np.float32)
    if bundle["model"] in ("H1", "H2"):
        from tensorflow.keras.models import load_model
        extractor = load_model(args.model_dir / "extractor.keras", compile=False)
        x = extractor.predict(x[:, None, :], verbose=0)
    elif bundle["model"] in ("H3", "H4"):
        x = hidden_features(bundle["extractor"], x)
    else:
        x = bundle["extractor"].transform(x)
    probability = bundle["classifier"].predict_proba(x)[:, 1]
    result = pd.DataFrame({"probability_attack": probability,
                           "predicted_label": (probability >= bundle["threshold"]).astype(int)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
