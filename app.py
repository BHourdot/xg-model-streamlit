import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("🧠 Calculateur xG (Expected Goals) - Modèle ML")

# Charger le modèle
try:
    model, feature_names = joblib.load("xG_model.pkl")
except:
    st.error("Erreur : Modèle non trouvé. Ajoutez xG_model.pkl au dossier.")
    st.stop()

# Saisie utilisateur
x = st.slider("Position X (depuis la ligne de but, en mètres)", 0.0, 120.0, 105.0)
y = st.slider("Position Y (largeur du terrain, centre = 40)", 0.0, 80.0, 40.0)
shot_type = st.selectbox("Type de tir", ["Foot", "Header", "Weak Foot"])
situation = st.selectbox("Situation", ["Open Play", "Set Piece", "Counter"])

# Calcul des features
distance = np.sqrt((120 - x)**2 + (y - 40)**2)
angle = np.arctan2(7.32 * (120 - x), ((120 - x)**2 + (y - 40)**2 - (7.32 / 2)**2))
angle_deg = np.degrees(angle)

# Préparer les données pour le modèle
input_dict = {
    "distance": distance,
    "angle_deg": angle_deg,
    "shot_type_Foot": 0,
    "shot_type_Header": 0,
    "shot_type_Weak Foot": 0,
    "situation_Counter": 0,
    "situation_Open Play": 0,
    "situation_Set Piece": 0,
}
input_dict[f"shot_type_{shot_type}"] = 1
input_dict[f"situation_{situation}"] = 1

# Assurer toutes les colonnes
for col in feature_names:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])[feature_names]

# Prédiction
xg = model.predict_proba(input_df)[:, 1][0]
st.metric("xG estimé", f"{xg:.3f}")
