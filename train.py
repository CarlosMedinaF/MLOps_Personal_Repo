"""
Entrenamiento de Regresión Logística sobre UCI Adult (Census Income) con MLflow Tracking.

Uso:
    python train.py                      # valores por defecto
    python train.py --C 0.1 --penalty l1 --class-weight balanced
    python train.py --help
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_PATH = Path(__file__).resolve().parent / "data" / "adult.csv"
TARGET = "income"


def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena una Regresión Logística para predecir ingreso >50K (UCI Adult) "
                    "y registra la corrida en MLflow.")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización; valores bajos = más regularización (default: 1.0)")
    p.add_argument("--penalty", choices=["l1", "l2"], default="l2",
                   help="Tipo de regularización (default: l2)")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="Peso de clases; 'balanced' compensa el desbalance de la clase >50K (default: none)")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Máximo de iteraciones del solver (default: 1000)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Proporción del conjunto de prueba (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria para split y modelo (default: 42)")
    p.add_argument("--experiment-name", default="adult_income_logreg_experimentos",
                   help="Nombre del experimento en MLflow (default: adult_income_logreg_experimentos)")
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000",
                   help="URI del servidor de MLflow (default: http://127.0.0.1:5000)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path=DATA_PATH):
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_csv(URL, header=None, names=COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean_data(df):
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    df = df.replace("?", np.nan)
    for c in ["workclass", "occupation", "native-country"]:
        df[c] = df[c].fillna("Unknown")
    df = df.drop_duplicates().reset_index(drop=True)
    df["log_capital_gain"] = np.log1p(df["capital-gain"])
    df["log_capital_loss"] = np.log1p(df["capital-loss"])
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df.drop(columns=["capital-gain", "capital-loss", "education", "fnlwgt"])


def split_data(df, test_size, seed):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_pipeline(X, C, penalty, class_weight, max_iter, seed):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = LogisticRegression(C=C, penalty=penalty, solver="liblinear",
                             class_weight=None if class_weight == "none" else class_weight,
                             max_iter=max_iter, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model, X, y, prefix):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
    }


def save_artifacts(model, X_test, y_test, metrics, params, out_dir):
    out_dir = Path(out_dir)
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))
    return [cm_path, metrics_path]


def main():
    args = parse_args()
    set_seed(args.seed)

    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    params = {
        "model": "LogisticRegression",
        "C": args.C,
        "penalty": args.penalty,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "solver": "liblinear",
        "test_size": args.test_size,
        "seed": args.seed,
    }
    run_name = f"logreg_C{args.C}_{args.penalty}_cw-{args.class_weight}_seed{args.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = build_pipeline(X_train, args.C, args.penalty, args.class_weight,
                               args.max_iter, args.seed)
        model.fit(X_train, y_train)

        metrics = {**evaluate(model, X_test, y_test, "test"),
                   **evaluate(model, X_train, y_train, "train")}
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        with tempfile.TemporaryDirectory() as tmp:
            for path in save_artifacts(model, X_test, y_test, metrics, params, tmp):
                mlflow.log_artifact(str(path))

        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

        print(f"\nRun: {run_name}  (run_id={run.info.run_id})")
        for k in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
            print(f"  {k:15s} {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
