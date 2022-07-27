import pandas as pd
from sklearn.pipeline import Pipeline  # type: ignore

from rossmann.model.pipeline.feature_extraction import (
    DataFrameFeatureExtractor,
    DateEncoder,
    DaySinceCompetitionOpenedExtractor,
    Promo2Extractor,
)


def make_feature_extractor(prepared_stores: pd.DataFrame) -> Pipeline:
    """Create a feature extractor pipeline.

    This is relatively classifier independent.

    Extracted features:
    * Day/Year/Month/DayOfYear: seasonal features
    * DaysSinceStart/Promo2: If Promo2 is at this day and how long this store
        is participating in the promotion
    * CompetitionOpenSinceDays/CompetitionOpen: if there is a competition open
    * CompetitionDistance: how far away is the competition
    * StoreType/Assortment: store characteristics
    * HolidayGroup: Dummy group of bundesland
    """
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
