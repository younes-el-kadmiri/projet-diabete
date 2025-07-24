import joblib
import os
import sys
import streamlit as st
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.preprocessing import scale_features


# Ajout du dossier parent au path si besoin (optionnel selon ton organisation)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Chemins vers les fichiers sauvegardés
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'kmeans_model.pkl'))

try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error(f"Scaler non trouvé au chemin : {SCALER_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Modèle non trouvé au chemin : {MODEL_PATH}")
    st.stop()

st.title("🩺 Prédiction du Risque de Diabète")

with st.form(key='diabete_form'):
    pregnancies = st.number_input("Nombre de grossesses", min_value=0, step=1)
    glucose = st.number_input("Taux de glucose", min_value=0, step=1)
    blood_pressure = st.number_input("Pression artérielle", min_value=0, step=1)
    skin_thickness = st.number_input("Épaisseur de la peau", min_value=0, step=1)
    insulin = st.number_input("Taux d'insuline", min_value=0, step=1)
    bmi = st.number_input("IMC", min_value=0.0, step=0.01, format="%.2f")
    dpf = st.number_input("Fonction de pedigree du diabète", min_value=0.0, step=0.001, format="%.3f")
    age = st.number_input("Âge", min_value=0, step=1)

    submit_button = st.form_submit_button(label='Prédire')
    reset_button = st.form_submit_button(label='Réinitialiser')

if submit_button:
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]], columns=features)

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Distance au centre du cluster prédit
    distances = model.transform(input_scaled)
    distance_to_center = distances[0][prediction]

    if prediction == 1:
        st.error(f"🚨 Le patient appartient au cluster à risque élevé (distance au centre : {distance_to_center:.3f})")
    else:
        st.success(f"✅ Le patient appartient au cluster à faible risque (distance au centre : {distance_to_center:.3f})")




if reset_button:
    st.experimental_rerun()
