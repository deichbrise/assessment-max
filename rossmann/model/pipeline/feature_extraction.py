from sklearn.base import TransformerMixin  # type: ignore
import pandas as pd
from rossmann.model.pipeline.execeptions import StoreNotFoundException
from typing import List, Text, cast


class DateEncoder(TransformerMixin):
    """Extracts time features from the index Date

    Expects a DataFrame with a datatimeindex Date

    """

    def fit(self, X: pd.DataFrame, y=None):
        self.start_data = min(X.index.get_level_values("Date"))
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        date: pd.DatetimeIndex = cast(
            pd.DatetimeIndex, X.index.get_level_values("Date")
        )
        return X.assign(
            Day=date.day,  # type: ignore
            Year=date.year,  # type: ignore
            Month=date.month,  # type: ignore
            DayOfYear=date.dayofyear,  # type: ignore
            DaysSinceStart=(date - self.start_data).days,
        )


class Promo2Extractor(TransformerMixin):
    KEEP_COLUMNS = ["Promo2", "PromoInterval", "Promo2Since"]

    def __init__(self, stores: pd.DataFrame) -> None:
        super().__init__()
        self.stores = stores[self.KEEP_COLUMNS]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_out = X.copy()
        X_out["Promo2"] = X.apply(
            lambda row: self.is_promo2_at_day(*row.name), axis=1
        )
        return X_out

    def is_promo2_at_day(self, storeId: int, date: pd.Timestamp) -> bool:
        if storeId not in self.stores.index.get_level_values(0):
            raise StoreNotFoundException(storeId)
        store_data = self.stores.loc[storeId]
        if not store_data["Promo2"]:
            return False
        return (
            date.month in store_data["PromoInterval"]
            and date > store_data["Promo2Since"]
        )


class DaySinceCompetitionOpenedExtractor(TransformerMixin):
    KEEP_COLUMNS = ["CompetitionOpenSince"]

    def __init__(self, stores: pd.DataFrame) -> None:
        super().__init__()
        self.stores = stores[self.KEEP_COLUMNS]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_out = X.copy()
        X_out["CompetitionOpenSinceDays"] = X.apply(
            lambda row: self.days_since_competition_opened(*row.name), axis=1
        )
        X_out["CompetitionOpen"] = X_out["CompetitionOpenSinceDays"] > 0
        return X_out

    def days_since_competition_opened(
        self, storeId: int, date: pd.Timestamp
    ) -> int:
        if storeId not in self.stores.index.get_level_values(0):
            raise StoreNotFoundException(storeId)
        store_data = self.stores.loc[storeId]
        if not store_data["CompetitionOpenSince"] or pd.isna(
            store_data["CompetitionOpenSince"]
        ):
            return 0
        return (date - store_data["CompetitionOpenSince"]).days


class DataFrameFeatureExtractor(TransformerMixin):
    def __init__(
        self, stores: pd.DataFrame, features: List[Text], id: Text
    ) -> None:
        super().__init__()
        self.stores = stores[features]
        self.id = id

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.join(self.stores, on=self.id, how="left")
