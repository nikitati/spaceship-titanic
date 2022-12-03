from pathlib import Path

import click
import pandas as pd


def compute_features(df: pd.DataFrame):
    df["PassengerGroup"] = df["PassengerId"].str.split("_", expand=True)[0]
    df_cabin = (
        df["Cabin"]
        .str.split("/", expand=True)
        .rename(columns={0: "CabinDeck", 1: "CabinNum", 2: "CabinSide"})
    )
    df = pd.concat(
        [
            df,
            df_cabin,
        ],
        axis=1,
    )
    df["CabinNum"] = df["CabinNum"].astype(float)
    df["TotalExpenses"] = df[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ].sum(axis=1, skipna=False, min_count=1)
    df["NoBill"] = df["TotalExpenses"] == 0
    group_size = df["PassengerGroup"].value_counts().rename("GroupSize")
    df = pd.merge(
        df, group_size, how="left", left_on="PassengerGroup", right_index=True
    )
    # df["SoloTraveler"] = df["GroupSize"] == 1
    return df


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option("--output-file", type=click.Path(path_type=Path))
def main(input_file: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file)
    df_features = compute_features(df)
    df_features.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
