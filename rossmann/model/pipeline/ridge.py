import argparse
from pathlib import Path
import logging
from pprint import pformat
from sklearn.compose import (  # type: ignore
    ColumnTransformer,
    TransformedTargetRegressor,
)
from sklearn.model_selection import (  # type: ignore
    GridSearchCV,
    train_test_split,
)

from sklearn.preprocessing import (  # type: ignore
    MinMaxScaler,
    PowerTransformer,
    OneHotEncoder,
)  # type: ignore
from sklearn.compose import make_column_selector
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.metrics import make_scorer, r2_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

import pandas as pd
from joblib import dump  # type: ignore

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
                "unskew",
                PowerTransformer(),
                ["CompetitionDistance", "CompetitionOpenSinceDays"],
            ),
            (
                "normnalize",
                MinMaxScaler(),
                [
                    "Day",
                    "Year",
                    "Month",
                    "DayOfYear",
                    "DayOfWeek",
                    "DaysSinceStart",
                ],
            ),
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
            (
                "regressor",
                Ridge(),
            ),
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
    params = {
        "regressor__regressor__alpha": [
            0.1,
            1,
            10,
        ],
    }
    rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
    grid_search = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring=rmspe_scorer,
    )
    pipeline = grid_search.fit(X_train, y_train)
    logger.info("GridSearchCV results:")
    logger.info(pformat(grid_search.cv_results_))

    logger.info("Training pipeline done.")
    logger.info(pformat(pipeline.cv_results_))
    return pipeline


def main(args: argparse.Namespace):
    logger.info(f"Running training with {args}")
    seed(args.seed)

    data_path = args.path
    out_path = data_path / "models" / f"ridge_{args.seed}"
    logger.info(f"Saving models/predictions/scores to {out_path}")
    try:
        out_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.warning(f"{out_path} already exists. Overwriting old results.")

    data = load_instances_csv(data_path / "train.csv")
    data = data.drop(columns=["Customers"])
    subset_top = TopStoreSelector(top_percent=0.1).fit_transform(data)
    assert subset_top.notna().all().all()
    train_data, eval_data = train_test_split(
        subset_top, test_size=0.1, random_state=args.seed
    )
    X_train, y_train = (
        train_data.drop(columns=["Sales"]),
        train_data["Sales"],
    )

    stores = load_stores_csv(data_path / "store.csv")
    prepared_stores = prepare_stores(data, stores)
    # Promo2Since is NaTa if Promo2 False.
    # That is ok, since this is only for lookup if Promo2 is True.
    assert prepared_stores.drop("Promo2Since", axis=1).notna().all().all()

    pipeline = train(X_train, y_train, prepared_stores)

    logger.info("Evaluating pipeline...")
    eval_data = {"train": train_data, "eval": eval_data}

    scores = evaluate(
        pipeline,
        eval_data,
        metrics={"rmspe": rmspe, "r2": r2_score},
    )
    logger.info(pformat(scores))

    dump(pipeline, out_path / "pipeline.pkl")

    X_test = load_instances_csv(data_path / "test.csv")

    assert X_test.notna().all().all()

    logger.info("Predicting on test data...")
    X_test_predicted = predict(pipeline, X_test.drop(columns=["Id"]))

    write_kaggle_evaluation_file(out_path, X_test_predicted, X_test["Id"])


def parse_args() -> argparse.Namespace:
    # todo add help text
    parser = argparse.ArgumentParser(
        """Train a Ridge regression model.
    1. Feature extraction pipeline: lookup from stores and extract date
        features.
    2. Feature transformation pipeline: transform features: normalize,
        log, etc.
    3. Regression pipeline: train a Ridge regression model.
    4. CrossValidation to find alpha  for the Ridge regression model."""
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
