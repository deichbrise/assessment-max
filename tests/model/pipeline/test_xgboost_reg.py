from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from pytest import fixture
from rossmann.model.data_loader import load_instances_csv, load_stores_csv
from rossmann.model.metrics import rmspe
from rossmann.model.pipeline.xgboost_reg import train
from rossmann.model.prepare_data import prepare_stores


@fixture(scope="session")
def data_path():
    return Path("data")


@fixture(scope="session")
def data(data_path):
    data = load_instances_csv(data_path / "train.csv").drop(
        columns=["Customers"]
    )
    train_data = data.sample(10000)
    eval_data = data.drop(train_data.index).sample(1000)

    eval_set = eval_data.drop(columns=["Sales"]), eval_data["Sales"]
    train_set = train_data.drop(columns=["Sales"]), train_data["Sales"]
    return (
        train_set,
        eval_set,
    )


@fixture(scope="session")
def stores(data_path, data):
    (X_train, y_train), _ = data
    stores = load_stores_csv(data_path / "store.csv")
    return prepare_stores(X_train, stores)


@fixture(scope="session")
def trained(data, stores):
    (X_train, y_train), _ = data

    pipeline = train(X_train, y_train, stores)

    return pipeline


def test_train_and_load(trained, data):
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        joblib.dump(trained, tmp / "pipeline.pkl")
        loaded = joblib.load(tmp / "pipeline.pkl")

    _, (X_eval, _) = data

    pred_load = loaded.predict(X_eval)
    pred = trained.predict(X_eval)
    assert (pred == pred_load).all()


def test_performance(trained, data):
    _, (X_eval, y_eval) = data
    pred = trained.predict(X_eval)
    assert rmspe(y_eval, pred) < 0.4
