import argparse
import csv
from pathlib import Path
import logging
from sklearn.compose import (  # type: ignore
    ColumnTransformer,
    TransformedTargetRegressor,
)

from sklearn.preprocessing import (  # type: ignore
    OneHotEncoder,
    PowerTransformer,
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
from rossmann.model.pipeline.train_utils import predict, seed
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

    target_transform = TransformedTargetRegressor(
        pipeline, transformer=PowerTransformer()
    )
    return target_transform


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

    train_data["IncludedInTraining"] = False
    train_data.loc[subset_train.index, "IncludedInTraining"] = True

    train_data_sampled = train_data.groupby("IncludedInTraining").sample(
        len(X_train)
    )
    logger.info(
        f"Evaluating on (sampled) train data (# {len(train_data_sampled)})"
    )
    y_train_pred = predict(pipeline, train_data_sampled)
    y_train_pred.name = "PredictedSales"
    train_data_sampled_pred = train_data_sampled.join(y_train_pred)
    scores = train_data_sampled_pred.groupby("IncludedInTraining").apply(
        lambda group: rmspe(group["Sales"], group["PredictedSales"])
    )

    logger.info(f"RMSPE on train data: {scores}")

    scores.to_csv(out_path / "training_scores.csv")
    train_data_sampled_pred.to_csv(out_path / "train_sampled_prediction.csv")

    dump(pipeline, out_path / "pipeline.pkl")

    X_test = load_instances_csv(data_path / "test.csv")

    assert X_test.notna().all().all()

    logger.info("Predicting on test data...")
    y_test_pred = predict(pipeline, X_test.drop(columns="Id"))
    y_test_pred_with_id = X_test[["Id"]].join(y_test_pred)
    y_test_pred_with_id.to_csv(
        out_path / "test_predictions.csv",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


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
