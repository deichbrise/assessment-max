import argparse
from pathlib import Path
import logging
from sklearn.compose import (  # type: ignore
    ColumnTransformer,
)
from sklearn.metrics import make_scorer, r2_score  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore

from sklearn.preprocessing import (  # type: ignore
    OneHotEncoder,
)  # type: ignore
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline  # type: ignore

import pandas as pd
from joblib import dump  # type: ignore
from xgboost import XGBRegressor  # type: ignore

from rossmann.model.pipeline import make_feature_extractor
from rossmann.model.metrics import rmspe
from rossmann.model.data_loader import load_instances_csv, load_stores_csv
from rossmann.model.pipeline.feature_extraction import OneVSAllBinarizer
from rossmann.model.pipeline.train_utils import (
    evaluate,
    predict,
    seed,
    write_kaggle_evaluation_file,
)
from rossmann.model.prepare_data import prepare_stores
from rossmann.model.pipeline.filter import TopStoreSelector


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_feature_transform():

    feature_transform = ColumnTransformer(
        [
            (
                "to_binary",
                OneVSAllBinarizer(not_target="no holiday"),
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
        ],
        remainder="passthrough",
    )
    return feature_transform


def make_pipeline(prepared_stores: pd.DataFrame) -> Pipeline:
    feature_extractor = make_feature_extractor(prepared_stores)
    feature_transform = make_feature_transform()
    # score get a negative sign  due to minimization behaviour
    # https://stackoverflow.com/a/27323356/3411517
    pipeline = Pipeline(
        [
            ("feature_extractor", feature_extractor),
            ("feature_transform", feature_transform),
            ("regressor", XGBRegressor(eval_metric=rmspe)),
        ]
    )
    return pipeline


def train(
    X_train: pd.DataFrame, y_train: pd.DataFrame, prepared_stores: pd.DataFrame
) -> Pipeline:
    """Trains a Ridge regression model and pipeline with the given data.

    Args:
        X_train (pd.DataFrame): _description_
        y_train (pd.DataFrame): _description_
        prepared_stores (pd.DataFrame): _description_

    Returns:
        Pipeline: _description_
    """
    logger.info(f"Training pipeline on {len(X_train)} instances...")
    pipeline = make_pipeline(prepared_stores)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=3,
        scoring=make_scorer(
            rmspe,
        ),
    )

    logger.info(f"Pre training cv (3) results in rmspe {scores}")

    pipeline = pipeline.fit(X_train, y_train)
    logger.info("Training pipeline done.")
    return pipeline


def main(args: argparse.Namespace):
    logger.info(f"Running training with {args}")
    seed(args.seed)

    data_path = args.path
    out_path = data_path / "models" / f"xgboost_{args.seed}"
    logger.info(f"Saving models/predictions/scores to {out_path}")
    try:
        out_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.warning(f"{out_path} already exists. Overwriting old results.")

    train_data = load_instances_csv(data_path / "train.csv")
    train_data = train_data.drop(columns=["Customers"])
    subset_train = TopStoreSelector(top_percent=0.1).fit_transform(train_data)
    # subset_train = train_data.sample(49742)
    assert subset_train.notna().all().all()

    X_train, y_train = (
        subset_train.drop(columns=["Sales"]),
        subset_train["Sales"],
    )

    stores = load_stores_csv(data_path / "store.csv")
    prepared_stores = prepare_stores(train_data, stores)
    # Promo2Since is NaTa if Promo2 False.
    # That is ok, since this is only for lookup if Promo2 is True.
    assert prepared_stores.drop("Promo2Since", axis=1).notna().all().all()

    pipeline = train(X_train, y_train, prepared_stores)

    logger.info("Evaluating pipeline...")
    eval_data = {
        "train": subset_train,
        "train_not_included": train_data.drop(subset_train.index).sample(
            len(subset_train)
        ),
    }

    scores = evaluate(
        pipeline,
        eval_data,
        metrics={"rmspe": rmspe, "r2": r2_score},
    )
    logger.info(scores)

    dump(pipeline, out_path / "pipeline.pkl")

    X_test = load_instances_csv(data_path / "test.csv")

    assert X_test.notna().all().all()

    logger.info("Predicting on test data...")
    X_test_predicted = predict(pipeline, X_test.drop(columns=["Id"]))

    write_kaggle_evaluation_file(out_path, X_test_predicted, X_test["Id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """Train a xgboost regression model.
    1. Feature extraction pipeline: lookup from stores and extract date
        features.
    2. Feature transformation pipeline:  to dummy  binary features.
    3. Regression pipeline: train a xgboost regression model."""
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
