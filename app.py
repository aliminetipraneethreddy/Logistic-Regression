import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Analysis", layout="centered")
st.title("📊 Customer Churn Prediction Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

df = load_data()

# -----------------------------
# Data Preprocessing
# -----------------------------
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

# -----------------------------
# Train-Test Split & Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(
    y_test, y_pred, target_names=["No Churn", "Churn"]
)
conf_matrix = confusion_matrix(y_test, y_pred)

# -----------------------------
# Predict for All Customers
# -----------------------------
df["Predicted_Churn"] = model.predict(X)
churn_count = (df["Predicted_Churn"] == 1).sum()
stay_count = (df["Predicted_Churn"] == 0).sum()

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader("📌 Model Performance")

st.metric("Accuracy", f"{accuracy:.2f}")

st.subheader("📄 Classification Report")
st.text(class_report)

st.subheader("📊 Confusion Matrix")
cm_df = pd.DataFrame(
    conf_matrix,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)
st.dataframe(cm_df)

# -----------------------------
# Churn Summary
# -----------------------------
st.subheader("🚨 Churn Prediction Summary")

st.metric("Total Customers", len(df))
st.metric("Customers Likely to Leave", churn_count)
st.metric("Customers Likely to Stay", stay_count)

# -----------------------------
# Sample Predictions
# -----------------------------
st.subheader("📋 Sample Customer Predictions")

st.dataframe(
    df[["tenure", "MonthlyCharges", "TotalCharges", "Predicted_Churn"]]
    .replace({"Predicted_Churn": {0: "No", 1: "Yes"}})
    .head(10)
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Logistic Regression | Streamlit | Customer Churn Analysis")
