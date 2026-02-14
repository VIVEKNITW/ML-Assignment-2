
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ML Classification App",
    layout="wide"
)

st.title("Machine Learning Classification Web Application")
st.markdown("Upload a test CSV file and evaluate different ML models.")

# -------------------- MODEL PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Logistic Regression": os.path.join(BASE_DIR, "models/logistic.pkl"),
    "Decision Tree": os.path.join(BASE_DIR, "models/decision_tree.pkl"),
    "KNN": os.path.join(BASE_DIR, "models/knn.pkl"),
    "Naive Bayes": os.path.join(BASE_DIR, "models/naive_bayes.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "models/random_forest.pkl"),
    "XGBoost": os.path.join(BASE_DIR, "models/xgboost.pkl"),
}

# -------------------- MODEL SELECTION --------------------
model_name = st.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV format only)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Error reading CSV file.")
        st.stop()

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # -------------------- CHECK TARGET --------------------
    if "target" not in data.columns:
        st.error("The uploaded dataset must contain a 'target' column for evaluation.")
        st.stop()

    X = data.drop("target", axis=1)
    y_true = data["target"]

    # -------------------- LOAD MODEL --------------------
    try:
        model = joblib.load(MODEL_PATHS[model_name])
    except Exception as e:
        st.error("Model file not found or failed to load.")
        st.stop()

    # -------------------- LOAD SCALER --------------------
    try:
        scaler = joblib.load("models/scaler.pkl")
        X = scaler.transform(X)
    except:
        st.warning("Scaler not found. Proceeding without scaling.")

    # -------------------- PREDICTION --------------------
    y_pred = model.predict(X)

    # AUC if available
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None

    # -------------------- METRICS --------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(acc, 4))
    col2.metric("Precision", round(prec, 4))
    col3.metric("Recall", round(rec, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1, 4))
    col5.metric("MCC", round(mcc, 4))

    if auc is not None:
        col6.metric("AUC Score", round(auc, 4))
    else:
        col6.metric("AUC Score", "Not Available")

    # -------------------- CONFUSION MATRIX --------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

    # -------------------- PREDICTIONS DISPLAY --------------------
    st.subheader("Predictions Output")

    output_df = data.copy()
    output_df["Predicted"] = y_pred

    st.dataframe(output_df.head())