from rossmann.server.model.store import Store
from typing import Dict, Union, Text
from dataclasses import asdict

from flask import current_app

from rossmann.model.pipeline.execeptions import StoreNotFoundException


def post(storeId: int, metadata: Dict[Text, Union[Text, int, bool]]):

    store = Store(Store=storeId, **metadata)  # type: ignore
    try:
        predicted_store = current_app.config["predictor"].predict(store)
        return asdict(predicted_store)
    except StoreNotFoundException as e:
        (
            f"Store storeId: {e.store} not found",
            404,
        )
