from dataclasses import dataclass
from typing import Text


@dataclass
class Store:
    Store: int
    DayOfWeek: int
    Date: Text  # TODO change / validate
    Open: bool
    Promo: bool
    StateHoliday: bool
    SchoolHoliday: bool


@dataclass
class PredictedStore:
    Store: int
    Date: Text
    PredictedSales: float
