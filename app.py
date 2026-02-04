import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


MODEL_DIR = "model"
DATA_PATH = "data/wdbc.data"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="ML Model Evaluation App", layout="wide")
st.title("ML Classification Model Evaluation")


def model_dumps_exist():
    if not os.path.exists(MODEL_DIR):
        return False
    files = os.listdir(MODEL_DIR)
    return (
        "scaler.pkl" in files and
        "features.pkl" in files and
        any(f.endswith(".pkl") and f not in ["scaler.pkl", "features.pkl"] for f in files)
    )


def delete_model_dumps():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)


def train_and_dump_models():
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

    df = pd.read_csv(DATA_PATH, header=None, names=columns)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    for name, model in models.items():
        model.fit(X_scaled, y)
        joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ', '_')}.pkl")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/features.pkl")


@st.cache_resource
def load_saved_models():
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl") and file not in ["scaler.pkl", "features.pkl"]:
            name = file.replace("_", " ").replace(".pkl", "")
            models[name] = joblib.load(os.path.join(MODEL_DIR, file))

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

    return models, scaler, features


st.subheader("Model Management")

if st.button("Rebuild Models", type="primary"):
    with st.spinner("Rebuilding models..."):
        delete_model_dumps()
        train_and_dump_models()
        st.cache_resource.clear()
        st.success("Models rebuilt successfully. Please upload test data.")
        time.sleep(.5)
    st.rerun()


if not model_dumps_exist():
    with st.spinner("Initializing models..."):
        train_and_dump_models()
    st.success("Models initialized successfully!")

models, scaler, feature_columns = load_saved_models()


st.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "CSV must contain 30 feature columns + diagnosis (M/B)",
    type="csv"
)

if uploaded_file is None:
    st.stop()

test_df = pd.read_csv(uploaded_file)

if "diagnosis" not in test_df.columns:
    st.error("CSV must contain a 'diagnosis' column.")
    st.stop()

X_test = test_df[feature_columns]
y_true = test_df["diagnosis"].map({"M": 1, "B": 0})
X_test_scaled = scaler.transform(X_test)


st.subheader("Select Model")
selected_model_name = st.selectbox("Choose model", list(models.keys()))
model = models[selected_model_name]


if st.button("Evaluate Model", type="primary"):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [
            accuracy_score(y_true, y_pred),
            roc_auc_score(y_true, y_prob),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            matthews_corrcoef(y_true, y_pred)
        ]
    })
    st.table(metrics_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).transpose()
    st.dataframe(report_df)
