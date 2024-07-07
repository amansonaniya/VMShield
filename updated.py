import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Function to load and preprocess the dataset
def load_data(file):
    df = pd.read_csv(file)
    X = df.drop("intrusion", axis=1)
    y = df["intrusion"]
    # Encode categorical labels if needed
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# Function to train and evaluate the selected machine learning algorithm
def train_evaluate_model(X_train, X_test, y_train, y_test, selected_algorithm):
    if selected_algorithm == "Random Forest":
        model = RandomForestClassifier()
    elif selected_algorithm == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif selected_algorithm == "SVM":
        model = SVC()
    else:
        return None

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, conf_matrix, end_time - start_time


# Streamlit UI
st.title("Machine Learning Model Performance")
st.write(
    "Upload a CSV file and select a machine learning algorithm to evaluate its performance."
)

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    st.write("CSV file uploaded successfully.")
    algorithm = st.selectbox(
        "Select Machine Learning Algorithm",
        ["Random Forest", "K-Nearest Neighbors", "SVM"],
    )
    submit_button = st.button("Submit")

    if submit_button:
        st.write("Training and evaluating the selected algorithm...")
        X, y = load_data(file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        accuracy, precision, recall, conf_matrix, execution_time = train_evaluate_model(
            X_train, X_test, y_train, y_test, algorithm
        )

        # Display performance metrics
        st.write("### Performance Metrics")
        st.write(f"Algorithm: {algorithm}")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"Execution Time: {execution_time:.4f} seconds")

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("Actual Labels")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
