import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("heart.csv")  # replace with your path

# Selected 5 features for demo
features = ["Age", "Sex", "RestingBP", "Cholesterol", "MaxHR"]

X = df[features]
y = df["HeartDisease"]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Heart Disease Model Accuracy:", accuracy)

# Save model, scaler, and imputer together
save_path = os.path.join(os.path.dirname(__file__), "heart_model.joblib")
joblib.dump({"model": clf, "scaler": scaler, "imputer": imputer}, save_path)
print("Saved heart_model.joblib")
