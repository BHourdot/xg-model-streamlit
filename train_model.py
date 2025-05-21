import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Génération de données factices
np.random.seed(42)
n_samples = 200
data = pd.DataFrame({
    "x": np.random.uniform(80, 120, n_samples),
    "y": np.random.uniform(0, 80, n_samples),
    "shot_type": np.random.choice(["Foot", "Header", "Weak Foot"], n_samples),
    "situation": np.random.choice(["Open Play", "Set Piece", "Counter"], n_samples),
    "is_goal": np.random.binomial(1, 0.1, n_samples)
})

data["distance"] = np.sqrt((120 - data["x"])**2 + (40 - data["y"])**2)
data["angle_deg"] = np.degrees(np.arctan2(7.32 * (120 - data["x"]), ((120 - data["x"])**2 + (data["y"] - 40)**2 - (7.32 / 2)**2)))

encoded = pd.get_dummies(data[["shot_type", "situation"]])
X = pd.concat([data[["distance", "angle_deg"]], encoded], axis=1)
y = data["is_goal"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump((model, X.columns.tolist()), "xG_model.pkl")
print("✅ Modèle entraîné et sauvegardé sous xG_model.pkl")
