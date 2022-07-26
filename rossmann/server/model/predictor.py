from dataclasses import asdict
from pathlib import Path
import joblib  # type: ignore
import pandas as pd

from sklearn.pipeline import Pipeline  # type: ignore

from rossmann.server.model.store import Store, PredictedStore


class StorePredictor:
    def __init__(self, pipeline_file: Path) -> None:
        self.pipeline = self.__load_pipeline(pipeline_file)

    @staticmethod
    def __load_pipeline(pipeline_file: Path) -> Pipeline:
        return joblib.load(pipeline_file)

    def predict(self, store: Store) -> PredictedStore:
        store_df = pd.DataFrame([asdict(store)])
        store_df["Date"] = pd.to_datetime(store_df["Date"])
        store_df.set_index(["Store", "Date"], inplace=True)

        predicted_sales = self.pipeline.predict(store_df)
        predicted = PredictedStore(
            Store=store.Store,
            Date=store.Date,
            PredictedSales=predicted_sales.round().astype(int).item(),
        )
        return predicted

    def __call__(self, store: Store) -> PredictedStore:
        return self.predict(store)
