import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_base_models(data_path="data/wdbc.data", model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)

    columns = [
        "id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
        "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    df = pd.read_csv(data_path, header=None, names=columns)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    trained_models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    for name, model in trained_models.items():
        model.fit(X_scaled, y)

        filename = name.replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(model_dir, filename))

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, "features.pkl"))

    return trained_models, scaler, X.columns.tolist()
