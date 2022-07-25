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


def make_feature_transform():
    feature_transform = ColumnTransformer(
        [
            ("unskew", PowerTransformer(), ["CompetitionDistance"]),
            (
                "to_binary",
                FunctionTransformer(lambda x: x != "no holiday"),
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
