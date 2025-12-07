import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("diabetes.csv")  # make sure this CSV is in the models folder

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Diabetes Accuracy:", acc)

# Save model
save_path = os.path.join(os.path.dirname(__file__), "diabetes_model.joblib")

joblib.dump({"model": model, "scaler": scaler, "imputer": imputer}, save_path)
print("Saved diabetes_model.joblib")
