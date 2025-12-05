# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Iris ML API")

MODEL_PATH = Path("models/model.joblib")
model = None


# 📋 Schéma d'entrée pour /predict
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# 🔄 Chargement du modèle au démarrage
@app.on_event("startup")
def load_model():
    """Charge le modèle au démarrage de l'application."""
    global model

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    else:
        raise RuntimeError(
            f"❌ Modèle introuvable : {MODEL_PATH}. Lance train.py d'abord."
        )


# 🩺 Endpoint de santé
@app.get("/health")
def health():
    """Vérifie que l'API fonctionne."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


# 🔮 Endpoint de prédiction
@app.post("/predict")
def predict(iris: IrisInput):
    """
    Prend les mesures d'une fleur et renvoie la classe prédite.
    Classes :
        - 0 : Setosa
        - 1 : Versicolor
        - 2 : Virginica
    """

    if model is None:
        return {"error": "Model not loaded"}

    # 🧱 Préparer les données pour le modèle
    df = pd.DataFrame(
        [[
            iris.sepal_length,
            iris.sepal_width,
            iris.petal_length,
            iris.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ],
    )

    # 🔍 Faire la prédiction
    pred = model.predict(df)[0]

    return {"prediction": int(pred)}
