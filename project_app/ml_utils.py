import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------------
# Age group mapping
# -----------------------------------
AGE_MAP = {
    0: "Child",
    1: "Adult",
    2: "Senior"
}

# -----------------------------------
# Feature Engineering
# -----------------------------------
def create_feature_matrix(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    height_weight_ratio = X[:, 0] / X[:, 1]
    pitch_height_ratio = X[:, 2] / X[:, 0]

    return np.column_stack([
        X_poly,
        height_weight_ratio.reshape(-1, 1),
        pitch_height_ratio.reshape(-1, 1)
    ])

# -----------------------------------
# Load trained ML models
# -----------------------------------
gender_model = joblib.load('gender_model.pkl')
age_model = joblib.load('age_model.pkl')
scaler = joblib.load('scaler.pkl')

# -----------------------------------
# Predict Gender + Age + Accuracy
# -----------------------------------
def predict_gender_age(height, weight, voice_pitch, bmi):
    X = np.array([[height, weight, voice_pitch, bmi]])
    X_scaled = scaler.transform(X)
    X_enhanced = create_feature_matrix(X_scaled)

    # Predicted class
    predicted_gender = gender_model.predict(X_enhanced)[0]
    predicted_age_num = age_model.predict(X_enhanced)[0]

    # Predicted probability (max confidence)
    gender_conf = np.max(gender_model.predict_proba(X_enhanced))
    age_conf = np.max(age_model.predict_proba(X_enhanced))

    # Convert age number → label
    predicted_age_label = AGE_MAP[predicted_age_num]

    return {
        "gender": "Male" if predicted_gender == 1 else "Female",
        "gender_confidence": round(gender_conf * 100, 2),

        "age_group": predicted_age_label,
        "age_confidence": round(age_conf * 100, 2)
    }
