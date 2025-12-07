from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
diab = joblib.load("models/diabetes_model.joblib")
heart = joblib.load("models/heart_model.joblib")
mal = joblib.load("models/malaria_model.joblib")

# Helper function to encode heart disease categorical inputs
def encode_heart_data(form):
    # Mapping example (adjust if your model uses different encoding)
    sex_map = {"M": 1, "F": 0}
    cp_map = {"typical": 0, "atypical": 1, "non-anginal": 2, "asymptomatic": 3}
    restecg_map = {"normal": 0, "ST-T": 1, "hypertrophy": 2}
    exang_map = {"yes": 1, "no": 0}
    slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
    thal_map = {"normal": 1, "fixed_defect": 2, "reversible_defect": 3}

    data = [
        float(form["age"]),
        sex_map.get(form["sex"].lower(), 0),
        cp_map.get(form["cp"].lower(), 0),
        float(form["trestbps"]),
        float(form["chol"]),
        float(form["fbs"]),
        restecg_map.get(form["restecg"].lower(), 0),
        float(form["thalach"]),
        exang_map.get(form["exang"].lower(), 0),
        float(form["oldpeak"]),
        slope_map.get(form["slope"].lower(), 0),
        float(form["ca"]),
        thal_map.get(form["thal"].lower(), 1)
    ]
    return np.array(data).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    data = [
        float(request.form["pregnancies"]),
        float(request.form["glucose"]),
        float(request.form["bp"]),
        float(request.form["skin"]),
        float(request.form["insulin"]),
        float(request.form["bmi"]),
        float(request.form["dpf"]),
        float(request.form["age"])
    ]
    X = np.array(data).reshape(1, -1)
    X = diab["imputer"].transform(X)
    X = diab["scaler"].transform(X)
    prob = diab["model"].predict_proba(X)[0][1]
    pred = 1 if prob >= 0.5 else 0
    return render_template("result.html", disease="Diabetes", probability=prob, result=pred)

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    features = ["Age", "Sex", "RestingBP", "Cholesterol", "MaxHR"]
    
    # Read data from form
    data = [float(request.form[f]) for f in features]
    X = np.array(data).reshape(1, -1)

    # Transform and scale
    X = heart["imputer"].transform(X)
    X = heart["scaler"].transform(X)

    # Predict
    prob = heart["model"].predict_proba(X)[0][1]
    pred = 1 if prob >= 0.5 else 0

    return render_template("result.html", disease="Heart Disease", probability=prob, result=pred)

@app.route("/predict_malaria", methods=["POST"])
def predict_malaria():
    data = [
        float(request.form["fever"]),
        float(request.form["headache"]),
        float(request.form["chills"]),
        float(request.form["vomit"]),
        float(request.form["sweats"])
    ]
    X = np.array(data).reshape(1, -1)
    X = mal["scaler"].transform(X)
    prob = mal["model"].predict_proba(X)[0][1]
    pred = 1 if prob >= 0.5 else 0
    return render_template("result.html", disease="Malaria", probability=prob, result=pred)

if __name__ == "__main__":
    app.run(debug=True)
