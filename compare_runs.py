"""Exporta la tabla de corridas del experimento a runs_comparacion.csv y la imprime ordenada por test_f1."""
import argparse

import mlflow
import pandas as pd

p = argparse.ArgumentParser(description="Exporta las corridas de un experimento de MLflow a CSV.")
p.add_argument("--experiment-name", default="adult_income_logreg_experimentos", help="Nombre del experimento")
p.add_argument("--tracking-uri", default="http://127.0.0.1:5000", help="URI del servidor de MLflow")
p.add_argument("--output", default="runs_comparacion.csv", help="Archivo CSV de salida")
args = p.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
runs = mlflow.search_runs(experiment_names=[args.experiment_name], order_by=["metrics.test_f1 DESC"])

cols = ["tags.mlflow.runName", "params.C", "params.penalty", "params.class_weight", "params.seed",
        "metrics.test_accuracy", "metrics.test_precision", "metrics.test_recall", "metrics.test_f1",
        "metrics.test_roc_auc", "metrics.train_f1", "run_id"]
table = runs[[c for c in cols if c in runs.columns]]
table.columns = [c.split(".")[-1] for c in table.columns]
table.to_csv(args.output, index=False)

pd.set_option("display.width", 250, "display.max_columns", None, "display.precision", 4)
print(table.drop(columns="run_id").to_string(index=False))
print(f"\n{len(table)} corridas exportadas a {args.output}")
