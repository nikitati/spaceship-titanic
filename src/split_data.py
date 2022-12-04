from pathlib import Path

import click
from dvc.api import params_show
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@click.command()
@click.option(
    "--input-data",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default="data/raw/train.csv",
)
@click.option("--output-dir", type=click.Path(path_type=Path), default="data/datasets/")
def main(input_data: Path, output_dir: Path):
    params = params_show(stages="split-data")["split-data"]
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_data)
    groups = df["PassengerId"].str.split("_", expand=True)[0]
    idx_train, idx_val = next(
        GroupShuffleSplit(
            n_splits=1,
            random_state=params["random_state"],
            test_size=params["val_size"],
        ).split(df, groups=groups)
    )
    df_train = df.iloc[idx_train]
    df_val = df.iloc[idx_val]
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)


if __name__ == "__main__":
    main()
