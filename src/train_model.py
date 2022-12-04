from pathlib import Path

import click
import pandas as pd
import numpy as np
from dvc.api import params_show
from mlem.api import save
from functools import partial, update_wrapper
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

# (feature name, is_categorical_tree_split)
ORDINAL_FEATURES = [
    ("HomePlanet", True),
    ("Destination", True),
    ("CabinDeck", False),
    ("CabinSide", True),
]
NUMERICAL_FEATURES = [
    ("RoomService", False),
    ("FoodCourt", False),
    ("ShoppingMall", False),
    ("Spa", False),
    ("VRDeck", False),
    ("TotalExpenses", False),
    ("CabinNum", False),
    ("Age", False),
]
BINARY_FEATURES = [
    ("CryoSleep", False),
    ("VIP", False),
    ("NoBill", False),
    ("SoloTraveler", False),
]
ALL_FEATURES = [
    *ORDINAL_FEATURES,
    *NUMERICAL_FEATURES,
    *BINARY_FEATURES,
]
FEATURE_NAMES, IS_CAT_FEATURE = zip(*ALL_FEATURES)


def create_model():
    model_params = params_show(stages="train")["train"]
    ordinal_feature_names, _ = zip(*ORDINAL_FEATURES)
    remainder_features = [fn for fn in FEATURE_NAMES if fn not in ordinal_feature_names]
    column_transformer = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            ordinal_feature_names,
        ),
        (
            "passthrough",
            remainder_features,
        ),
        remainder="drop",
    )
    hgbc = HistGradientBoostingClassifier(
        categorical_features=list(IS_CAT_FEATURE),
        **model_params,
    )
    classifier = make_pipeline(
        column_transformer,
        hgbc,
    )
    return classifier


def train_model(df_input: pd.DataFrame):
    classifier = create_model()
    X = df_input
    y = df_input["Transported"]
    classifier.fit(X, y)
    return classifier


def create_submission(model, X):
    return pd.DataFrame(
        {
            "PassengerId": X["PassengerId"],
            "Transported": model.predict(X),
        }
    )


@click.command()
@click.option(
    "--training-data",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option("--model-path", type=click.Path(path_type=Path))
def main(training_data: Path, model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(training_data)
    model = train_model(df_train)
    sample_data = df_train[["PassengerId", *FEATURE_NAMES]].head(n=10)
    save(
        model,
        model_path,
        sample_data=sample_data,
    )
    save(
        update_wrapper(partial(create_submission, model), create_submission),
        model_path.with_name(model_path.name + "_lambda"),
        sample_data=sample_data,
    )


if __name__ == "__main__":
    main()
