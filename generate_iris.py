# generate_iris.py

from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path

def main():
    # Charger le dataset Iris depuis scikit-learn
    iris = load_iris(as_frame=True)
    df = iris.frame   # DataFrame avec features + target

    # Créer le dossier data/ s'il n'existe pas
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Sauvegarder en CSV
    csv_path = data_dir / "iris.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Dataset iris sauvegardé dans {csv_path}")

if __name__ == "__main__":
    main()
