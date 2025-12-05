# train.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

DATA_PATH = Path("data/iris.csv")
MODELS_DIR = Path("models")


def main():
    # 1️⃣ Charger les données
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}. "
            "Lance generate_iris.py d'abord."
        )

    df = pd.read_csv(DATA_PATH)

    # Séparer features (X) et target (y)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2️⃣ Configurer MLflow
    mlflow.set_experiment("iris-logreg")

    with mlflow.start_run():
        # Définir les hyperparamètres
        C = 1.0
        max_iter = 200

        # Logger les paramètres
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        # 3️⃣ Entraîner le modèle
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)

        # 4️⃣ Évaluer le modèle
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"🎯 Accuracy sur le test: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

        # 5️⃣ Sauvegarder le modèle localement
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "model.joblib"
        joblib.dump(model, model_path)

        print(f"💾 Modèle sauvegardé dans {model_path}")

        # 6️⃣ Logger le modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
