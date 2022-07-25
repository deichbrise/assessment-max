import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore

from rossmann.model.pipeline.feature_extraction import (
    DataFrameFeatureExtractor,
    DateEncoder,
    DaySinceCompetitionOpenedExtractor,
    Promo2Extractor,
)


def make_feature_extractor(prepared_stores: pd.DataFrame) -> Pipeline:
    feature_extraction = Pipeline(
        [
            ("date_features", DateEncoder()),
            ("promo2_features", Promo2Extractor(prepared_stores)),
            (
                "competition_open_features",
                DaySinceCompetitionOpenedExtractor(prepared_stores),
            ),
            (
                "store_features",
                DataFrameFeatureExtractor(
                    prepared_stores,
                    features=[
                        "StoreType",
                        "Assortment",
                        "CompetitionDistance",
                        "HolidayGroup",
                    ],
                    id="Store",
                ),
            ),
        ]
    )
    return feature_extraction
