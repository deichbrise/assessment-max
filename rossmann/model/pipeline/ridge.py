import argparse
from pathlib import Path
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.preprocessing import (  # type: ignore
    PowerTransformer,
    OneHotEncoder,
    FunctionTransformer,
)  # type: ignore
from sklearn.compose import make_column_selector
from rossmann.model.metrics import rmspe
from sklearn.linear_model import RidgeCV  # type: ignore
from sklearn.metrics import make_scorer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
import pandas as pd

from rossmann.model.pipeline import make_feature_extractor
from joblib import dump  # type: ignore
from rossmann.model.data_loader import load_instances_csv, load_stores_csv
from rossmann.model.prepare_data import prepare_stores
from rossmann.model.pipeline.filter import TopStoreSelector


def to_binary_holidays(categorical_holidays: pd.Series) -> pd.Series:
    return categorical_holidays != "no holiday"


def make_feature_transform():

    feature_transform = ColumnTransformer(
        [
            ("unskew", PowerTransformer(), ["CompetitionDistance"]),
            (
                "to_binary",
                FunctionTransformer(to_binary_holidays),
                ["StateHoliday"],
            ),
            (
                "one_hot_encode_bin",
                OneHotEncoder(sparse=False, drop="if_binary"),
                make_column_selector(dtype_include=["bool"]),
            ),
            (
                "one_hot_encode_Assortment",
                OneHotEncoder(
                    sparse=False, categories=[["basic", "extra", "extended"]]
                ),
                ["Assortment"],
            ),
            (
                "one_hot_encode_StoreType",
                OneHotEncoder(sparse=False, categories=[["a", "b", "c", "d"]]),
                ["StoreType"],
            ),
            (
                "one_hot_encode_HolidayGroup",
                OneHotEncoder(
                    sparse=False,
                    categories=[[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                ),
                ["HolidayGroup"],
            ),
            ("drop_binary", "drop", ["Customers"]),
        ],
        remainder="passthrough",
    )
    return feature_transform


def make_pipeline(prepared_stores: pd.DataFrame) -> Pipeline:
    feature_extractor = make_feature_extractor(prepared_stores)
    feature_transform = make_feature_transform()
    # score get a negative sign  due to minimization behaviour
    # https://stackoverflow.com/a/27323356/3411517
    rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
    pipeline = Pipeline(
        [
            ("feature_extractor", feature_extractor),
            ("feature_transform", feature_transform),
            (
                "regressor",
                RidgeCV(scoring=rmspe_scorer, cv=5, alphas=[0.1, 1.0, 10.0]),
            ),
        ]
    )

    return pipeline


def train(data_path: Path) -> Pipeline:
    # todo split into train and main
    # todo report progress / scores

    train = load_instances_csv(data_path / "train.csv")
    subset_train = TopStoreSelector(top_percent=0.1).fit_transform(train)
    assert subset_train.notna().all().all()
    X_train, y_train = (
        subset_train.drop(columns=["Sales"]),
        subset_train["Sales"],
    )

    stores = load_stores_csv(data_path / "store.csv")
    prepared_stores = prepare_stores(train, stores)

    # Promo2Since is NaTa if Promo2 False.
    # That is ok, since this is only for lookup if Promo2 is True.
    assert prepared_stores.drop("Promo2Since", axis=1).notna().all().all()

    pipeline = make_pipeline(prepared_stores)
    pipeline = pipeline.fit(X_train, y_train)

    dump(pipeline, data_path / "ridge_model.pkl")


def parse_args() -> argparse.Namespace:
    # todo add help text
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = train(args.path)
