import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# Feature Engineering
# -----------------------------
def create_feature_matrix(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    height_weight_ratio = X[:, 0] / X[:, 1]
    pitch_height_ratio = X[:, 2] / X[:, 0]

    return np.column_stack(
        [X_poly, height_weight_ratio.reshape(1, -1), pitch_height_ratio.reshape(1, -1)]
    )

# -----------------------------
# Load ML models
# -----------------------------
gender_model = joblib.load('gender_model.pkl')
age_model = joblib.load('age_model.pkl')
scaler = joblib.load('scaler.pkl')

# -----------------------------
# Predict Function
# -----------------------------
def predict_gender_age(height, weight, voice_pitch, bmi):
    X = np.array([[height, weight, voice_pitch, bmi]])
    X_scaled = scaler.transform(X)
    X_enhanced = create_feature_matrix(X_scaled)

    predicted_gender = gender_model.predict(X_enhanced)
    predicted_age = age_model.predict(X_enhanced)

    return predicted_gender[0], predicted_age[0]
