# disease_prediction_system
A web app that lets a user (or clinician) enter patient features (age, sex, blood pressure, glucose, symptoms, etc.) and returns a predicted probability / risk for one or more diseases (Diabetes, Heart disease, Malaria). Each disease will be handled by a trained ML model (tabular models). You can later extend to image-based malaria detection if you want.

Project goals / objectives

Build supervised ML models to predict presence/absence (or risk) of Diabetes, Heart disease and Malaria from clinical/symptom data.

Provide a simple web UI where users can input features and get predictions + explanation (feature importances).

Evaluate models using appropriate metrics (accuracy, precision, recall, ROC AUC).

Package trained models and demonstrate live inference in the app.

Scope & assumptions (keeps it manageable)

Use tabular datasets (clinical/symptom features). Avoid image ML unless you have time/compute.

Build one model per disease (easier to explain & implement).

Use classical ML (Logistic Regression, RandomForest, XGBoost if comfortable).

Frontend: simple HTML/CSS (Bootstrap) + Flask backend (Python).

Suggested datasets (well-known, easy to use)

Diabetes: Pima Indians Diabetes Dataset (UCI / Kaggle) — features like pregnancies, glucose, blood pressure, BMI, age.

Heart Disease: Cleveland Heart Disease dataset (UCI) — includes age, sex, cholesterol, resting blood pressure, etc.

Malaria (tabular): If you don’t use images, either create a symptom-based dataset (fever, chills, headache, travel_history, platelet_count) OR find clinical malaria datasets. For ND, you can simulate a small labeled symptom dataset if none is easily available.

(If you want image-based malaria later: there are public datasets of blood smear images, but that requires CNNs and more compute.)

High-level system architecture

Data & model training (offline)

Python (pandas, scikit-learn, xgboost)

Train & evaluate models

Save models with joblib or pickle.

Web application (runtime)

Backend: Flask (Python) loads saved models, serves prediction endpoints.

Frontend: HTML + Bootstrap forms to capture input features; show results.

Optionally add explanation: show feature importances or SHAP values (if advanced).

Deployment (optional): host locally or on Heroku / PythonAnywhere.

Step-by-step implementation plan
1) Environment & packages

Python 3.8+ with:

pandas, numpy

scikit-learn

joblib

flask

matplotlib / seaborn (for EDA)

optionally xgboost, shap

Install:

pip install pandas numpy scikit-learn joblib flask matplotlib
# optional:
pip install xgboost shap

2) Data preprocessing (common steps)

Load CSV.

Handle missing values (imputation or drop).

Feature scaling where needed (StandardScaler for algorithms that need it).

Encode categorical variables (one-hot or label encoding).

Split into train/test (e.g., 80/20).

Optionally use cross-validation and hyperparameter tuning (GridSearchCV).

3) Model training example (Diabetes with Pima dataset)
