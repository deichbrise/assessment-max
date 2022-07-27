from sklearn.base import TransformerMixin  # type: ignore
import pandas as pd


class TopStoreSelector(TransformerMixin):
    """Select the instances from stores that make top_percent of the sales"""

    def __init__(self, top_percent: float) -> None:
        self.top_percent = top_percent

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Select the  top performing stores that have to 10% of the sales
        total_sales = X["Sales"].sum()
        sales_per_store = (
            X.groupby(by="Store")
            .agg({"Sales": "sum"})
            .sort_values(by="Sales", ascending=False)
        )
        sales_per_store["accumulated_sales"] = sales_per_store[
            "Sales"
        ].cumsum()  # accumulated sales
        top_stores_sales = sales_per_store.query(
            f"accumulated_sales <= {total_sales} * {self.top_percent}"
        )

        subset_train = X.join(
            top_stores_sales.drop(columns=["accumulated_sales", "Sales"]),
            on="Store",
            how="inner",
        )

        return subset_train
