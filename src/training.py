"""Train hybrid classifiers without fitting preprocessing on validation/test data."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"H1": "LSTM + XGBoost", "H2": "LSTM + Random Forest",
          "H3": "MLP + Extra Trees", "H4": "MLP + Histogram Gradient Boosting",
          "H5": "PCA + Random Forest"}
EXCLUDED = {"id", "label", "attack_cat"}


def feature_frame(df):
    return df.drop(columns=list(EXCLUDED), errors="ignore").replace([np.inf, -np.inf], np.nan)


def group_split(df, seed=42):
    """Keep identical feature vectors together, even if duplicated in the CSV."""
    groups = pd.util.hash_pandas_object(feature_frame(df), index=False)
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(df, groups=groups))
    for idx in (fit, val):
        if df.iloc[idx].label.nunique() != 2:
            raise ValueError("Both classes must occur in training and validation.")
    return fit, val


def make_preprocessor(x, k):
    cats = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [c for c in x if c not in cats]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))])
    columns = ColumnTransformer([("numeric", numeric, nums), ("categorical", categorical, cats)])
    return Pipeline([("columns", columns), ("select", SelectKBest(f_classif, k=k))])


def hidden_features(mlp, x):
    out = x
    for weights, bias in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        out = np.maximum(out @ weights + bias, 0)
    return np.asarray(out, dtype=np.float32)


def train_mlp(x, y, xv, yv, args):
    net = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", batch_size=args.batch_size,
        learning_rate_init=.001, random_state=args.seed, early_stopping=False, shuffle=True)
    history = {key: [] for key in ("loss", "val_loss", "accuracy", "val_accuracy")}
    best, stale, snapshot = float("inf"), 0, None
    for epoch in range(args.epochs):
        net.partial_fit(x, y, classes=np.array([0, 1]))
        pt, pv = net.predict_proba(x)[:, 1], net.predict_proba(xv)[:, 1]
        for key, value in [("loss", log_loss(y, pt)), ("val_loss", log_loss(yv, pv)),
                           ("accuracy", accuracy_score(y, pt >= .5)), ("val_accuracy", accuracy_score(yv, pv >= .5))]:
            history[key].append(float(value))
        print(f"  Epoch {epoch+1}: validation loss={history['val_loss'][-1]:.4f}", flush=True)
        if history["val_loss"][-1] < best - 1e-5:
            best, stale, snapshot = history["val_loss"][-1], 0, copy.deepcopy(net)
        else:
            stale += 1
            if stale >= args.patience:
                break
    return snapshot, history


def train_lstm(x, y, xv, yv, args):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    inputs = tf.keras.Input(shape=(1, x.shape[1]))
    h = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    h = tf.keras.layers.Dropout(.3)(h)
    h = tf.keras.layers.LSTM(32)(h)
    features = tf.keras.layers.Dense(32, activation="relu", name="features")(h)
    prediction = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    model = tf.keras.Model(inputs, prediction)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x[:, None, :], y, validation_data=(xv[:, None, :], yv),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)])
    extractor = tf.keras.Model(inputs, features)
    return model, extractor, history.history


def savefig(folder, filename):
    plt.tight_layout()
    plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plots(folder, key, df, y, proba, history, importance, feature_kind, curve_kind):
    title = f"{key}: {MODELS[key]}"
    counts = df.label.value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(6, 4)); plt.bar(["Normal", "Attack"], counts, color=["steelblue", "tomato"])
    plt.ylabel("Records"); plt.title(title + " — input training distribution")
    savefig(folder, "01_class_distribution.png")
    plt.figure(figsize=(10, 5))
    df.attack_cat.fillna("Unknown").value_counts().plot.bar(color="steelblue")
    plt.ylabel("Records"); plt.title("Attack categories — training CSV (not input features)")
    plt.xticks(rotation=45, ha="right"); savefig(folder, "02_attack_types.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, train_key, val_key, label in [(axes[0], "loss", "val_loss", "Log loss"), (axes[1], "accuracy", "val_accuracy", "Accuracy")]:
        ax.plot(history[train_key], label="Fit"); ax.plot(history[val_key], label="Validation")
        ax.set_xlabel(curve_kind); ax.set_ylabel(label); ax.legend()
    fig.suptitle(title + " — " + ("RF stages" if key == "H5" else "feature extractor"))
    savefig(folder, "03_loss_accuracy.png")
    pred = (proba >= .5).astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4)); plt.imshow(matrix, cmap="Oranges")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() * .55 else "black")
    plt.xticks([0, 1], ["Normal", "Attack"]); plt.yticks([0, 1], ["Normal", "Attack"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); savefig(folder, "04_confusion_matrix.png")
    order = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 4)); plt.bar(np.arange(len(order)), importance[order], color="darkorange")
    plt.xticks(np.arange(len(order)), [f"{feature_kind}_{i}" for i in order], rotation=90)
    plt.title("Validation permutation importance (AP)" if key == "H4" else "Classifier importance — transformed features")
    plt.ylabel("Importance"); savefig(folder, "05_feature_importance.png")
    precision, recall, _ = precision_recall_curve(y, proba)
    plt.figure(figsize=(6, 4)); plt.plot(recall, precision, label=f"AP={average_precision_score(y, proba):.4f}")
    plt.axhline(float(np.mean(y)), color="gray", linestyle="--", label="Attack prevalence")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title(title)
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    savefig(folder, "06_precision_recall_curve.png")
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(6, 4)); plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y, proba):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.title(title); savefig(folder, "07_roc_curve.png")


def run(key, args):
    # Fail before reading the dataset if optional libraries are missing.
    if key in ("H1", "H2"):
        import tensorflow  # noqa: F401
    if key == "H1":
        import xgboost  # noqa: F401
    start = time.perf_counter()
    train_path, test_path = args.data_dir / "UNSW_NB15_training-set.csv", args.data_dir / "UNSW_NB15_testing-set.csv"
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for df in (train, test):
        if not {"label", "attack_cat"}.issubset(df): raise ValueError("label and attack_cat columns are required")
        if set(df.label.unique()) != {0, 1}: raise ValueError("Expected binary labels 0 and 1")
    if args.smoke:
        train, _ = train_test_split(train, train_size=min(4000, len(train)-1), stratify=train.label, random_state=args.seed)
        test, _ = train_test_split(test, train_size=min(2000, len(test)-1), stratify=test.label, random_state=args.seed)
    fit_idx, val_idx = group_split(train, args.seed)
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    x, xv, xt = feature_frame(fit), feature_frame(val), feature_frame(test)
    if set(x.columns) != set(xt.columns): raise ValueError("Train/test feature columns differ")
    xv, xt = xv[x.columns], xt[x.columns]
    prep = make_preprocessor(x, args.features)
    y, yv, yt = fit.label.to_numpy(), val.label.to_numpy(), test.label.to_numpy()
    x = prep.fit_transform(x, y).astype(np.float32)
    xv, xt = prep.transform(xv).astype(np.float32), prep.transform(xt).astype(np.float32)
    history, extractor, deep = None, None, None
    print(f"{key} {MODELS[key]}: fit={len(y)}, val={len(yv)}, test={len(yt)}", flush=True)
    if key in ("H3", "H4"):
        extractor, history = train_mlp(x, y, xv, yv, args)
        xf, vf, tf = [hidden_features(extractor, data) for data in (x, xv, xt)]
        feature_kind = "MLP"
    elif key in ("H1", "H2"):
        deep, extractor, history = train_lstm(x, y, xv, yv, args)
        xf, vf, tf = [extractor.predict(data[:, None, :], batch_size=args.batch_size, verbose=0) for data in (x, xv, xt)]
        feature_kind = "LSTM"
    else:
        extractor = PCA(n_components=min(16, x.shape[1]), random_state=args.seed)
        xf = extractor.fit_transform(x); vf, tf = extractor.transform(xv), extractor.transform(xt)
        feature_kind = "PC"
    tree_options = dict(n_estimators=args.trees, max_depth=18, min_samples_leaf=2,
        random_state=args.seed, n_jobs=args.jobs, class_weight="balanced")
    if key == "H1":
        from xgboost import XGBClassifier
        classifier = XGBClassifier(n_estimators=args.trees, max_depth=6, learning_rate=.05,
            tree_method="hist", eval_metric="logloss", random_state=args.seed, n_jobs=args.jobs)
    elif key == "H3": classifier = ExtraTreesClassifier(**tree_options)
    elif key == "H4": classifier = HistGradientBoostingClassifier(max_iter=args.trees,
        max_leaf_nodes=31, early_stopping=False, random_state=args.seed, class_weight="balanced")
    else: classifier = RandomForestClassifier(**tree_options)
    if key == "H5":
        history = {k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        classifier.set_params(warm_start=True, class_weight={0:float(weights[0]),1:float(weights[1])})
        stages = sorted(set([max(1, args.trees*i//4) for i in range(1, 5)]))
        for n in stages:
            classifier.set_params(n_estimators=n); classifier.fit(xf, y)
            a, b = classifier.predict_proba(xf)[:, 1], classifier.predict_proba(vf)[:, 1]
            for k, value in [("loss",log_loss(y,a)),("val_loss",log_loss(yv,b)),("accuracy",accuracy_score(y,a>=.5)),("val_accuracy",accuracy_score(yv,b>=.5))]:history[k].append(float(value))
    else: classifier.fit(xf, y)
    training_seconds = time.perf_counter() - start
    proba = classifier.predict_proba(tf)[:, 1]; pred = (proba >= .5).astype(int)
    if key == "H4":
        rng = np.random.default_rng(args.seed); ix = rng.choice(len(yv), min(1000,len(yv)), replace=False)
        importance = permutation_importance(classifier, vf[ix], yv[ix], scoring="average_precision", n_repeats=2, random_state=args.seed, n_jobs=1).importances_mean
    else: importance = classifier.feature_importances_
    hashes_train = pd.util.hash_pandas_object(feature_frame(train), index=False)
    hashes_test = pd.util.hash_pandas_object(feature_frame(test), index=False)
    metrics = dict(model=key, architecture=MODELS[key], mode="smoke" if args.smoke else "full",
        train_csv_rows=len(train), fit_rows=len(y), validation_rows=len(yv), test_rows=len(yt),
        accuracy=accuracy_score(yt,pred), precision_attack=precision_score(yt,pred,zero_division=0),
        recall_attack=recall_score(yt,pred,zero_division=0), f1_attack=f1_score(yt,pred,zero_division=0),
        f1_weighted=f1_score(yt,pred,average="weighted"), roc_auc=roc_auc_score(yt,proba),
        average_precision=average_precision_score(yt,proba), training_seconds=training_seconds,
        threshold=.5, seed=args.seed, epochs=args.epochs, trees=args.trees,
        features=args.features, batch_size=args.batch_size, patience=args.patience,
        selected_features=prep.get_feature_names_out().tolist(),
        test_rows_with_features_seen_in_train=int(hashes_test.isin(hashes_train).sum()),
        history=history, python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in ["numpy","pandas","scikit-learn","matplotlib","joblib"]},
        data_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [train_path,test_path]})
    root = args.output_dir / "smoke" if args.smoke else args.output_dir
    folder, models = root / f"image{key}", root / "models" / key
    folder.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    plots(folder,key,train,yt,proba,history,importance,feature_kind,"Tree-count stage (25%, 50%, 75%, 100%)" if key=="H5" else "Epoch (zero-based)")
    (folder/f"metrics_{key}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    report = classification_report(yt,pred,target_names=["Normal","Attack"],zero_division=0)
    text = f"{key}: {MODELS[key]}\nMode: {metrics['mode']}\nThreshold: 0.5\n"
    text += "\n".join(f"{k}: {metrics[k]:.6f}" for k in ["accuracy","precision_attack","recall_attack","f1_attack","f1_weighted","roc_auc","average_precision","training_seconds"])
    text += f"\n\n{report}\nTest rows with feature vectors seen in train: {metrics['test_rows_with_features_seen_in_train']}\n"
    text += "Original official split preserved; overlap limits generalization claims.\n"
    (folder/f"results_{key}.txt").write_text(text,encoding="utf-8")
    (folder/"ORIGINE.md").write_text(
        f"# {key} — résultats recalculés\n\nMode : {metrics['mode']}. Voir metrics_{key}.json pour les paramètres, versions et empreintes des données.\n",
        encoding="utf-8")
    bundle={"preprocessor":prep,"classifier":classifier,"extractor":extractor if deep is None else None,
            "feature_columns":feature_frame(fit).columns.tolist(),"model":key,"threshold":.5}
    joblib.dump(bundle,models/"pipeline.joblib",compress=3)
    if deep is not None:
        deep.save(models/"lstm.keras"); extractor.save(models/"extractor.keras")
    print(f"{key} complete: accuracy={metrics['accuracy']:.4f}, F1 attack={metrics['f1_attack']:.4f}; saved {folder}",flush=True)
    return metrics


def main(default_model=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",nargs="+",choices=list(MODELS),default=[default_model] if default_model else ["H3","H4","H5"])
    parser.add_argument("--data-dir",type=Path,default=ROOT/"DATA")
    parser.add_argument("--output-dir",type=Path,default=ROOT)
    parser.add_argument("--epochs",type=int,default=15); parser.add_argument("--trees",type=int,default=100)
    parser.add_argument("--features",type=int,default=25); parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--patience",type=int,default=4);parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--jobs",type=int,default=2);parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if min(args.epochs,args.trees,args.features,args.batch_size,args.jobs,args.patience)<1:parser.error("Numeric training parameters must be positive")
    with threadpool_limits(limits=args.jobs):
        for key in args.models:run(key,args)


if __name__=="__main__": main()
