import argparse
from pathlib import Path
import pandas as pd

from rossmann.model.data_loader import (
    load_instances_csv,
    load_stores_csv,
    extract_approx_bundesland,
)


def prepare_stores(
    train_instances: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Loads and preprocesses the stores data"""

    holiday_groups = extract_approx_bundesland(train_instances)
    stores = stores.join(holiday_groups, on="Store", how="left")

    stores["CompetitionDistance"].fillna(
        stores["CompetitionDistance"].median(), inplace=True
    )

    stores["CompetitionOpenSince"].fillna(
        stores["CompetitionOpenSince"].median(numeric_only=False), inplace=True
    )

    return stores


def main(args):
    train = load_instances_csv(args.path / "train.csv")
    stores = load_stores_csv(args.path / "store.csv")

    prepared_stores = prepare_stores(train, stores)
    prepared_stores.reset_index().to_feather(
        args.path / "prepared_stores.feather"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
