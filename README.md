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
### Setup
**Requirements**
* python>=3.8
* pip

### Development / Training the Model

```bash
# clone the repo

# Instarll the dev requirements: 
pip install -r requirements-dev.txt

# get the data.
# Make sure to place you credential in ~/.kaggle/kaggel.jshon or set the enironment varaibles 
# For more details see https://github.com/Kaggle/kaggle-api
kaggle competitions download -c rossmann-store-sales -p data
# unzip the data
unzip data/rossmann-store-sales.zip -d data/
# train the model 
python -m rossmann.model.pipeline.ridge data --seed 42
# run the (devleopment) server 
python -m rossmann.server.app
# open http://127.0.0.1:5000/sales/ui/ to tes
```