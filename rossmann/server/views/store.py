from typing import Dict, Union, Text


def post(storeId: int, metadata: Dict[Text, Union[Text, int, bool]]):

    print(f"Posting metadata for storeId: {metadata}")
    if storeId == 1:
        return (
            f"Store storeId: {storeId} not found",
            404,
        )
    return {"Store": storeId, "Date": "2015-09-17", "PredictedSales": 5263}
