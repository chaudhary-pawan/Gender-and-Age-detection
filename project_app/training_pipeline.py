import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, auc)

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    # Base features
    height = np.random.normal(170, 10, n_samples)
    weight = np.random.normal(70, 15, n_samples)
    voice_pitch = np.random.normal(150, 30, n_samples)
    
    # Add BMI as a derived feature
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    # Generate gender with slightly unbalanced distribution
    gender = np.random.binomial(1, 0.48, n_samples)
    
    # Age distribution
    age_probabilities = [0.2, 0.6, 0.2]
    age_group = np.random.choice(3, size=n_samples, p=age_probabilities)
    
    # Apply demographic effects
    # Gender effects
    height[gender == 0] -= np.random.normal(12, 2, sum(gender == 0))
    weight[gender == 0] -= np.random.normal(15, 3, sum(gender == 0))
    voice_pitch[gender == 0] += np.random.normal(80, 10, sum(gender == 0))
    
    # Age effects - Children
    child_mask = age_group == 0
    height[child_mask] = np.random.normal(135, 10, sum(child_mask))
    weight[child_mask] = np.random.normal(32, 8, sum(child_mask))
    voice_pitch[child_mask] += np.random.normal(50, 10, sum(child_mask))
    
    # Age effects - Seniors
    senior_mask = age_group == 2
    height[senior_mask] -= np.random.normal(3, 1, sum(senior_mask))
    weight[senior_mask] += np.random.normal(2, 1, sum(senior_mask))
    voice_pitch[senior_mask] -= np.random.normal(20, 5, sum(senior_mask))
    
    # Add correlated noise
    noise = np.random.normal(0, 2, n_samples)
    height += noise
    weight += noise * 1.5
    voice_pitch += noise * 3
    
    return pd.DataFrame({
        'height': height,
        'weight': weight,
        'voice_pitch': voice_pitch,
        'bmi': bmi,
        'gender': gender,
        'age_group': age_group
    })

def create_feature_matrix(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    height_weight_ratio = X[:, 0] / X[:, 1]
    pitch_height_ratio = X[:, 2] / X[:, 0]
    
    return np.column_stack([X_poly, height_weight_ratio.reshape(-1, 1),pitch_height_ratio.reshape(-1, 1)])

def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_feature_distributions(data):
    # Set the style for all plots
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    features = ['height', 'weight', 'voice_pitch', 'bmi']
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        sns.kdeplot(data=data[data['gender'] == 0][feature], label='Female', ax=ax)
        sns.kdeplot(data=data[data['gender'] == 1][feature], label='Male', ax=ax)
        ax.set_title(f'{feature.capitalize()} Distribution by Gender')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data):
    features = ['height', 'weight', 'voice_pitch', 'bmi']
    correlation_matrix = data[features].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.show()

def train_and_evaluate_models(X, y_gender, y_age):
    # Split data
    X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
        X, y_gender, y_age, test_size=0.2, random_state=42, stratify=y_gender
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create enhanced feature matrices
    X_train_enhanced = create_feature_matrix(X_train_scaled)
    X_test_enhanced = create_feature_matrix(X_test_scaled)

    # Initialize models
    gender_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                        min_samples_split=5, random_state=42)
    age_model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                        min_samples_split=5, random_state=42)

    # Train models
    gender_model.fit(X_train_enhanced, y_gender_train)
    age_model.fit(X_train_enhanced, y_age_train)

    # --------------------------
    # 🔥 SAVE MODELS AS PICKLE
    # --------------------------
    import joblib
    joblib.dump(gender_model, "gender_model.pkl")
    joblib.dump(age_model, "age_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Models saved as gender_model.pkl, age_model.pkl, scaler.pkl")

    # Predictions
    gender_pred = gender_model.predict(X_test_enhanced)
    age_pred = age_model.predict(X_test_enhanced)

    # Accuracy and reports
    return {
        'gender_accuracy': accuracy_score(y_gender_test, gender_pred),
        'age_accuracy': accuracy_score(y_age_test, age_pred),
        'gender_report': classification_report(y_gender_test, gender_pred),
        'age_report': classification_report(y_age_test, age_pred),
        'gender_model': gender_model,
        'age_model': age_model,
        'test_data': (X_test_enhanced, y_gender_test, y_age_test)
    }


def main():
    try:
        # Generate and prepare data
        data = generate_synthetic_data(2000)
        X = data[['height', 'weight', 'voice_pitch', 'bmi']]
        y_gender = data['gender']
        y_age = data['age_group']
        
        # Save data to a CSV file
        data.to_csv("synthetic_data.csv", index=False)
        print("Data saved to 'synthetic_data.csv'")
        
        # Plot initial data distributions
        plot_feature_distributions(data)
        plot_correlation_matrix(data)
        
        # Train and evaluate models
        results = train_and_evaluate_models(X, y_gender, y_age)
        
        # Print results
        print("Enhanced Model Results:")
        print("\nGender Classification:")
        print(f"Accuracy: {results['gender_accuracy']:.4f}")
        print("\nDetailed Gender Classification Report:")
        print(results['gender_report'])
        print("\nAge Group Classification:")
        print(f"Accuracy: {results['age_accuracy']:.4f}")
        print("\nDetailed Age Classification Report:")
        print(results['age_report'])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

