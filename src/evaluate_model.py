from pathlib import Path
import json

import click
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from mlem.api import load


def evaluate(X, y_true, classifier, pred_path):
    y_pred_ps = classifier.predict_proba(X)[:, 1]
    y_pred = y_pred_ps >= 0.5
    pd.DataFrame({
        "PassengerId": X["PassengerId"],
        "actual": y_true,
        "prediction": y_pred,
    }).to_csv(pred_path)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_ps),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


@click.command()
@click.option("--test-data", type=click.Path(path_type=Path))
@click.option("--model-path", type=click.Path(path_type=Path))
def main(test_data: Path, model_path: Path):
    df = pd.read_csv(test_data)
    classifier = load(model_path)
    metrics = evaluate(df, df["Transported"], classifier, Path("data/predictions") / test_data.name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
