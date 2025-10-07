# Application de prédiction de churn

Cette application fournit :

- un jeu de données synthétique (`data/churn_data.csv`) pour entraîner un modèle de churn ;
- un script d'entraînement en Python pur qui ajuste un modèle de régression logistique ;
- une API REST (FastAPI) pour exposer la prédiction ;
- une interface utilisateur Streamlit pour saisir les caractéristiques d'un client.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entraînement du modèle

```bash
python -m app.training
```

Le modèle et les métriques sont sauvegardés dans le dossier `models/`.

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

### Exemple d'appel

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5,
        "TotalCharges": 402.5
      }'
```

## Lancer l'interface Streamlit

```bash
streamlit run streamlit_app.py
```

L'application permet d'utiliser directement le modèle local ou d'appeler l'API FastAPI.
