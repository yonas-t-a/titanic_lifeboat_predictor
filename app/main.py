from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI()

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

def load_models():
    m_path = os.path.join(BASE_DIR, "models")
    model_files = {
        "Logistic Gate": "logistic_gate.pkl",
        "Random Forest": "random_forest_gate.pkl",
        "SVM Gate": "svm_gate.pkl",
        "XGBoost Gate": "xgboost_gate.pkl"
    }
    
    loaded = {}
    for label, filename in model_files.items():
        full_path = os.path.join(m_path, filename)
        if os.path.exists(full_path):
            try:
                loaded[label] = joblib.load(full_path)
                print(f"✅ Successfully Loaded: {label}")
            except Exception as e:
                print(f"⚠️ Error loading {label}: {e}")
        else:
            print(f"❌ File Missing: {full_path}")
    return loaded

models = load_models()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pclass: int = Form(...),
    sex_male: int = Form(...),
    age: float = Form(...),
    fare: float = Form(...)
):
    # 1. Define Column Order (Must match your Training Notebook X_train.columns)
    column_order = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    
    data = {
        'Pclass': [int(pclass)],
        'Age': [float(age)],
        'SibSp': [0],
        'Parch': [0],
        'Fare': [float(fare)],
        'Sex_male': [int(sex_male)],
        'Embarked_Q': [0],
        'Embarked_S': [1]
    }
    
    # Create DataFrame and enforce order
    input_df = pd.DataFrame(data)[column_order]

    gate_results = []
    
    for name, model in models.items():
        try:
            # FIX FOR XGBOOST: Convert to NumPy array to ignore feature name mismatch
            # This forces the model to use values based on position only.
            input_values = input_df.values if name == "XGBoost Gate" else input_df
            
            # Special handling for SVM if it wasn't trained with probability=True
            if name == "SVM Gate" and not hasattr(model, "predict_proba"):
                decision = model.decision_function(input_df)[0]
                prob = 1 / (1 + np.exp(-decision))
            else:
                # Get probabilities
                prob_array = model.predict_proba(input_values)
                prob = prob_array[0][1]
            
            gate_results.append({
                "name": name,
                "prob": round(float(prob) * 100, 1),
                "survived": prob > 0.5
            })
        except Exception as e:
            print(f"❌ Error predicting with {name}: {e}")
            gate_results.append({
                "name": name,
                "prob": 0.0,
                "survived": False,
                "error": True
            })

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "results": gate_results,
        "form_data": {"pclass": pclass, "sex_male": sex_male, "age": age}
    })