from __future__ import annotations

from typing import Dict

import streamlit as st

from app.prediction import churn_model

st.set_page_config(page_title="Churn Prediction", page_icon="📉")
st.title("📉 Application de prédiction de churn")
st.write(
    """
    Cette interface permet de saisir les informations d'un client et d'obtenir une prédiction de churn.
    Vous pouvez appeler l'API FastAPI ou utiliser le modèle local directement.
    """
)

use_api = st.checkbox("Utiliser l'API", value=False)
api_url = st.text_input("URL de l'API", value="http://localhost:8000/predict", disabled=not use_api)

options: Dict[str, list[str]] = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

with st.form("prediction_form"):
    gender = st.selectbox("Genre", options["gender"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partenaire", options["Partner"])
    dependents = st.selectbox("Personnes à charge", options["Dependents"])
    tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Service téléphonique", options["PhoneService"])
    multiple_lines = st.selectbox("Lignes multiples", options["MultipleLines"])
    internet_service = st.selectbox("Service internet", options["InternetService"])
    online_security = st.selectbox("Sécurité en ligne", options["OnlineSecurity"])
    online_backup = st.selectbox("Sauvegarde en ligne", options["OnlineBackup"])
    device_protection = st.selectbox("Protection des appareils", options["DeviceProtection"])
    tech_support = st.selectbox("Support technique", options["TechSupport"])
    streaming_tv = st.selectbox("Streaming TV", options["StreamingTV"])
    streaming_movies = st.selectbox("Streaming Films", options["StreamingMovies"])
    contract = st.selectbox("Contrat", options["Contract"])
    paperless_billing = st.selectbox("Facturation électronique", options["PaperlessBilling"])
    payment_method = st.selectbox("Méthode de paiement", options["PaymentMethod"])
    monthly_charges = st.number_input("Facturation mensuelle", min_value=0.0, value=70.0)
    total_charges = st.number_input("Facturation totale", min_value=0.0, value=tenure * monthly_charges)

    submitted = st.form_submit_button("Prédire")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    if use_api:
        import requests

        try:
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erreur lors de l'appel API: {exc}")
        else:
            st.success(f"Probabilité de churn: {result['probability']:.2%}")
            st.write(f"Prédiction: **{result['prediction']}**")
    else:
        result = churn_model.predict(payload)
        st.success(f"Probabilité de churn: {result['probability']:.2%}")
        st.write(f"Prédiction: **{result['prediction']}**")
