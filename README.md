# data-science-task

Mit dieser Aufgabe wollen wir gerne sehen, wie Du Probleme mit Code löst. Es geht weder
darum das beste Modell zu finden noch darum eine fertige App zu entwickeln.

## Aufgaben

1. Trainiere ein Modell für die Daten aus <https://www.kaggle.com/c/rossmann-store-sales/overview>.
   Wähle aus den ca. 3000 Stores die Teilmenge aus, die die obersten 10% der Gesamt-Sales-Menge in
   den Trainings-Daten ausmachen. Trainiere ein Modell für die Vorhersage der täglichen Sales pro
   Store für diese Teilmenge.

2. Überprüfe die Vorhersageleistung deines Modells auf den Testdaten mit dem *root mean square
   percentage error* (RSMPE) und einer weiteren Metrik Deiner Wahl. Der RSMPE sollte <= 0.4 sein,
   ansonsten ist die Performance aber nicht weiter wichtig.

3. Binde dein Modell in eine REST API über die Route `/sales/store/<store-id>` ein. Die Route 
   sollte einen POST-Request mit dem Body der Form

   ```js
   {
       DayOfWeek: 4,
       Date: "2015-09-17",
       Open: true,
       Promo: true,
       StateHoliday: false, 
       SchoolHoliday: false
   }
   ```

   entgegenehmen und eine Antwort der Form

    ```js
    {
       Store: 1,
       Date: "2015-09-17",
       PredictedSales: 5263
    }
    ```

    zurückgeben.

    Für Stores, die das Modell nicht kennt, sollte 404 zurückgegeben werden.

4. Dokumentiere deine Lösung so, dass Andere in der Lage sind, das Modell neu zu trainieren
   und die REST API lokal zu starten. Das kannst Du hier direkt im README machen (Englisch oder
   Deutsch). Checke alles ins repository ein, was notwendig ist, um Reproduzierbarkeit
   sicherzustellen.

Die Aufgabe sollte vollständig in Python gelöst werden. Du kannst aber alle Libraries und Tools Deiner Wahl verwenden.
Deine vollständige Lösung sollte auf einem separaten Branch eingecheckt sein.
Stelle einen pull request gegen den 'main' Branch, sobald Du fertig bist.

## Dokumentation

### Repo Structure: 
```
├── README.md
├── conftest.py
├── data                   # contain the data and trained models
├── pyproject.toml
├── pytest.ini
├── requirements-dev.txt   # dev only dependencies, also installs requirements.txt
├── requirements.txt       # production only only
├── rossmann               # the actual softwre package
│   ├── model              # model and training code
│   └── server             # server definition
└── tests                  # Some tests
```

### Setup
**Requirements**
* python>=3.8


### Development / Training the Model

```bash
# Instarll the dev requirements: 
pip install -r requirements-dev.txt

pre-commit install 
# get the data.
# Make sure to place you credential in ~/.kaggle/kaggel.jshon or set the enironment varaibles 
# For more details see https://github.com/Kaggle/kaggle-api
kaggle competitions download -c rossmann-store-sales -p data
# unzip the data
unzip data/rossmann-store-sales.zip -d data/
# train the model 
python -m rossmann.model.pipeline.xgboost_reg data --seed 42
# run the (devleopment) server 
python -m rossmann.server.app
# open http://127.0.0.1:5000/sales/ui/ to the REST doc and swagger
```

Instead of training an  xgboost you can also train a ridge regressor pipline using: 
```
python -m rossmann.model.pipeline.ridge data --seed 42
```

In order to switch the model used by the server change the `rossmann/server/config/default.py` to 
```python
MODEL_PATH = "data/models/rdige_42/pipeline.pkl"
```

The general pattern is: `data/models/[ridge|xgboost]_[seed]/pipeline.pkl`


### Models / EDA
The have a look at the EDA and more details about the two implmented piplines have a look at the notbooks: 

* rossmann/model/EDA.ipynb
* rossmann/model/piplines.ipynb

respectively.

To run the 


### Training/Evaluation Results

| Regressor |  Split | rmspe |   r2  |
|:---------:|:------:|:-----:|:-----:|
|  XGBoost |  train | 0.009 | 0.973 |
|  XGBoost |  eval  | 0.006 | 0.968 |
|   Rdige   |  train | 0.054 | 0.711 |
|  Rdige |   eval | 0.051 | 0.721 |