# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Installer les dépendances système si besoin
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements et installer Python dependencies
COPY requirements-api.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-api.txt

# Copier le code de l'application et les modèles
COPY app ./app
COPY models ./models

# Exposer le port
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
