"""Compare only freshly generated results with matching evaluation splits."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    results = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(args.results_dir.glob("imageH*/metrics_H*.json"))]
    if not results:
        raise SystemExit("No results found. Train a model first.")
    keys = ["mode", "data_sha256", "seed", "fit_rows", "validation_rows", "test_rows", "threshold"]
    for r in results[1:]:
        if any(r[k] != results[0][k] for k in keys):
            raise SystemExit("Evaluation splits differ. Run models with matching data and seed.")
    columns = ["model", "architecture", "mode", "accuracy", "precision_attack", "recall_attack",
               "f1_attack", "f1_weighted", "average_precision", "roc_auc", "training_seconds"]
    table = pd.DataFrame([{k: r[k] for k in columns} for r in results])
    destination = args.results_dir / "comparison"
    destination.mkdir(exist_ok=True)
    table.to_csv(destination / "comparison.csv", index=False)
    table.set_index("model")[["accuracy", "f1_attack", "average_precision", "roc_auc"]].plot.bar(figsize=(10, 5), ylim=(0, 1))
    plt.title("UNSW-NB15 — observed results (" + results[0]["mode"] + ")")
    plt.ylabel("Score"); plt.xticks(rotation=0); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(destination / "comparison.png", dpi=150); plt.close()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
