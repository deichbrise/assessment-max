from typing import List, Text
import pandas as pd
from pathlib import Path

from sklearn.compose import make_column_selector  # type: ignore

state_holiday_name_map = {
    "a": "public",
    "b": "Easter holiday",
    "c": "Christmas",
    "0": "no holiday",
}

assortment_name_map = {"a": "basic", "b": "extra", "c": "extended"}

data_columns = ["Date"]

columns_types = {
    "Open": "boolean",
    "Promo": "boolean",
    "SchoolHoliday": "boolean",
    "StateHoliday": "category",
    "StoreType": "category",
    "Assortment": "category",
    "Promo2": "boolean",
}

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
]


def boolean2bool(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert all boolean columns to bool type."""
    columns = make_column_selector(dtype_include="boolean")(dataframe)
    dataframe[columns] = dataframe[columns].astype(bool)
    return dataframe


def load_instances_csv(file_path: Path) -> pd.DataFrame:
    """Loads the instances (train or test) csv file.

    Converts:
    *  StateHoliday -> categorical


    Args:
        file_path (Path): train.csv or test.csv from the data folder

    Returns:
        pd.DataFrame: dataset with index: Store (int), Date (datetime64[ns])
        and the following columns:
            * DayOfWeek (int64)
            * Sales (int64)
            * Customers (int64)
            * Open (bool)
            * Promo (bool)
            * StateHoliday (category)
            * SchoolHoliday (bool)
    """
    instances = pd.read_csv(
        file_path,
        index_col=["Store", "Date"],
        parse_dates=data_columns,
        dtype=columns_types,  # type: ignore
    )

    instances["Open"].fillna(value=True, inplace=True)
    cats = instances["StateHoliday"].cat
    instances["StateHoliday"] = cats.rename_categories(state_holiday_name_map)

    return boolean2bool(instances)


def load_stores_csv(file_path: Path) -> pd.DataFrame:
    """Load the stores csv file.

    Converts:
    *  Assortment -> categorical
    * [CompetitionOpenSinceYear (int),  CompetitionOpenSinceMonth (int)]
        -> CompetitionOpenSince (datetime64[ns])
    * [Promo2SinceWeek (int) Promo2SinceYear (int)]
        -> Promo2Since (datetime64[ns])
    * PromoInterval -> List[int]

    Args:
        file_path (Path): stores.csv from the data folder

    Returns:
        pd.DataFrame: dataset with index Store (int) the following columns:
            * StoreType (category)
            * Assortment (category)
            * CompetitionDistance (float64)
            * Promo2 (bool)
            * PromoInterval (object)
            * CompetitionOpenSince (datetime64[ns])
            * Promo2Since (datetime64[ns])
    """

    stores = pd.read_csv(
        file_path, index_col=["Store"], dtype=columns_types  # type: ignore
    )
    # convert to Assortment categorical
    stores["Assortment"] = stores["Assortment"].cat.rename_categories(
        assortment_name_map
    )

    # [CompetitionOpenSinceYear (int),  CompetitionOpenSinceMonth (int)]
    # -> CompetitionOpenSince (datetime64[ns])
    stores["CompetitionOpenSince"] = pd.to_datetime(
        stores[["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]]
        .rename(
            columns={
                "CompetitionOpenSinceYear": "year",
                "CompetitionOpenSinceMonth": "month",
            }
        )
        .assign(day=1)
    )
    stores = stores.drop(
        ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"], axis=1
    )

    # [Promo2SinceWeek (int) Promo2SinceYear (int)]
    # -> Promo2Since (datetime64[ns])
    # assuming monday of the first week
    assert (
        (stores["Promo2SinceWeek"].notna() & stores["Promo2SinceYear"].notna())
        == stores["Promo2"]
    ).all()
    stores_with_promo_date = stores.query(
        "Promo2SinceWeek.notna() and Promo2SinceYear.notna()"
    )
    formatted_date = (
        stores_with_promo_date["Promo2SinceYear"].astype(int).astype(str)
        + " "
        + stores_with_promo_date["Promo2SinceWeek"].astype(int).astype(str)
        + " 1"
    )
    promo_start_date = pd.to_datetime(
        formatted_date, format="%Y %W %w", errors="coerce"
    )
    promo_start_date.name = "Promo2Since"
    stores = stores.join(promo_start_date, on="Store").drop(
        ["Promo2SinceWeek", "Promo2SinceYear"], axis=1
    )

    # PromoInterval -> List[int]

    def month_str2list(month_names: List[Text]) -> List[int]:
        if month_names == [""]:
            return []
        return [months.index(m_name) + 1 for m_name in month_names]

    stores["PromoInterval"] = (
        stores["PromoInterval"]
        .fillna(value="")
        .str.split(",")
        .apply(month_str2list)
    )

    stores = boolean2bool(stores)
    return stores


def extract_approx_bundesland(instances: pd.DataFrame) -> pd.Series:
    """
    Extract for each store an approximate Bundesland aka HolidayGroup.

    The instances data frame must have:
    * Store as index interger index
    * Date as index datetime index
    * SchoolHoliday as boolean columns

    Args:
        instances: DataFrame of

    Returns:
        Series:
            of shape (n_stores,),
            Store as index
            type category
            name HolidayGroup
    """
    SchoolHoliday_per_store = instances.reset_index().pivot(
        index="Store", columns="Date", values="SchoolHoliday"
    )
    holiday_groups = SchoolHoliday_per_store.groupby(
        SchoolHoliday_per_store.columns.to_list()
    ).ngroup()
    holiday_groups.index.name = "Store"
    holiday_groups.name = "HolidayGroup"
    holiday_groups = holiday_groups.astype("category")
    return holiday_groups
