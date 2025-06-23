import shap
import pandas as pd
import joblib


model = joblib.load("model.pkl")
X_train = pd.read_csv("x_test.csv")  
explainer = shap.Explainer(model.predict, X_train)

def get_shap_values(input_df):
    return explainer(input_df)
