import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("synthetic_credit_data.csv")
x = df.drop("default", axis=1)
y = df["default"]

x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y,test_size=0.2, random_state = 42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))

])

pipeline.fit(x_train, y_train)

joblib.dump(pipeline, "model.pkl")
x_test.to_csv("x_test.csv", index=False)
