import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import os

# --- Configuration et Chargement ---
MODEL_PATH = "models/best_model.joblib"
CONFIG_PATH = "models/serving_config.json"
EPSILON = 1e-5

#1. Charger le pipeline et la config 
@st.cache_resource
def load_resources():
    #Charger le pipeline complet
    obj = joblib.load(MODEL_PATH)
    pipeline = obj["pipeline"]
    feature_names = obj["features"]
    
    #Charger le seuil métier
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    thr = config.get("decision_threshold", 0.5)
    
    return pipeline, feature_names, thr

pipeline, feature_names, decision_threshold = load_resources()

st.title("MLOps Demo : Évaluation du Risque de Défaut de Prêt")
st.caption(f"Modèle sélectionné: {pipeline['clf'].__class__.__name__} | Seuil de décision métier: {decision_threshold:.2f}")

#--- Collecte des entrées utilisateur ---
st.header("Entrez les informations du demandeur")

col1, col2 = st.columns(2)

with col1:
    income = st.slider("Revenu Annuel (income)", min_value=10000, max_value=200000, value=70000, step=1000)
    total_debt_outstanding = st.slider("Dette Totale (total_debt_outstanding)", min_value=1000, max_value=50000, value=15000, step=500)
    fico_score = st.slider("Score FICO", min_value=500, max_value=850, value=650, step=1)

with col2:
    years_employed = st.slider("Années d'emploi (years_employed)", min_value=0, max_value=20, value=5, step=1)
    credit_lines_outstanding = st.number_input("nombre de crédit en cours", min_value=1, max_value=15, value=5)
    loan_amt_outstanding = st.slider("Montant d'emprunt en cours", min_value=1000, max_value=50000, value=10000, step=500)
    
#--- Logique de prédiction ---
if st.button("Calculer le Risque de Défaut de paiement"):
    
    #3. Création des données brutes (DataFrame)
    data = {
        'income': [income],
        'total_debt_outstanding': [total_debt_outstanding],
        'fico_score': [fico_score],
        'years_employed': [years_employed],
        'credit_lines_outstanding': [credit_lines_outstanding],
        'loan_amt_outstanding': [loan_amt_outstanding],
    }
    input_df = pd.DataFrame(data)
    
    #4. Feature Engineering (Recréer la logique du Notebook)
    #NOTE: Si 'debt_income_variable' est dans feature_names, elle doit être recréée. 
    #si les features du modèle ne utilisent pas.
    
    try:
        X_final = input_df[feature_names]
    except KeyError as e:
        st.error(f"Erreur de Feature Engineering/Sélection. Colonne manquante: {e}. Vérifiez la cohérence entre load_data(mode=strict) et le modèle sérialisé.")
        st.stop()

    #5. Prédiction avec le pipeline
    proba_default = pipeline.predict_proba(X_final)[:, 1][0]
    prediction = (proba_default >= decision_threshold).astype(int)

    #--- Affichage des résultats ---
    st.markdown("---")
    st.subheader("Résultat du Modèle")
    
    st.metric("Probabilité de Défaut (PD)", f"{proba_default*100:.2f} %")
    
    if prediction == 1:
        st.error(f"🔴 DÉCISION : DÉFAUT (PD ≥ {decision_threshold:.2f})")
        st.write("Le risque de défaut de paiement est élevé selon le seuil métier. Prêt non recommandé.")
    else:
        st.success(f"🟢 DÉCISION : REMBOURSEMENT (PD < {decision_threshold:.2f})")
        st.write("Le risque de défaut de paiement est faible. Prêt recommandé.")