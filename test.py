import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.models import Sequential
from sklearn.ensemble import IsolationForest
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("preprocessed_dataset.csv")
y_data = data["intrusion"]
X_data = data.drop(labels=["intrusion"], axis=1)
X_data = np.array(X_data)
y_data = np.array(y_data)


X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.20, random_state=42
)


def lstm(X_train, y_train, X_test, y_test):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    from keras.utils import to_categorical

    y_train_encoded = to_categorical(y_train, num_classes=5)
    y_test_encoded = to_categorical(y_test, num_classes=5)

    y_train = y_train_encoded
    y_test = y_test_encoded

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(units=50))
    # output layer with softmax azctivation
    model.add(Dense(units=5, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    with st.spinner("Training LSTM model..."):
        history = model.fit(
            X_train, y_train, epochs=10, batch_size=5000, validation_split=0.2
        )

    st.write("Training completed!")

    test_results = model.evaluate(X_test, y_test, verbose=1)

    st.write(
        f"Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%"
    )

    # Plot accuracy vs epoch for train and test dataset
    st.write("Plot of accuracy vs epoch for train and test dataset")
    st.line_chart(
        {
            "Train Accuracy": history.history["accuracy"],
            "Test Accuracy": history.history["val_accuracy"],
        }
    )

    # Plot loss vs epoch for train and test dataset
    st.write("Plot of loss vs epoch for train and test dataset")
    st.line_chart(
        {
            "Train Loss": history.history["loss"],
            "Test Loss": history.history["val_loss"],
        }
    )


def isolation_forest(X_train, X_test):
    model = IsolationForest(random_state=42, contamination="auto")

    with st.spinner("Training Isolation Forest model..."):
        model.fit(X_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    y_pred_test_binary = np.where(y_pred_test == -1, 1, 0)

    classification_result = classification_report(
        y_test, y_pred_test_binary, output_dict=True
    )
    overall_metrics = {
        "Precision": classification_result["weighted avg"]["precision"],
        "Recall": classification_result["weighted avg"]["recall"],
        "F1-Score": classification_result["weighted avg"]["f1-score"],
        "Accuracy": accuracy_score(y_test, y_pred_test_binary),
    }

    st.write("Overall Metrics:")
    df_metrics = pd.DataFrame(overall_metrics, index=[0])
    st.table(df_metrics)

    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    st.pyplot(fig)


def svm(X_train, X_test):
    model = SVC(random_state=42)

    with st.spinner("Training SVM model..."):
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

    classification_result = classification_report(y_test, y_pred_test, output_dict=True)
    overall_metrics = {
        "Precision": classification_result["weighted avg"]["precision"],
        "Recall": classification_result["weighted avg"]["recall"],
        "F1-Score": classification_result["weighted avg"]["f1-score"],
        "Accuracy": accuracy_score(y_test, y_pred_test),
    }

    st.write("Overall Metrics (SVM):")
    df_metrics = pd.DataFrame(overall_metrics, index=[0])
    st.table(df_metrics)

    st.write("Confusion Matrix (SVM):")
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    st.pyplot(fig)


def knn(X_train, X_test):
    # Create the KNN model
    model = KNeighborsClassifier()

    # Fit the model
    with st.spinner("Training KNN model..."):
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

    # Evaluate the model
    classification_result = classification_report(y_test, y_pred_test, output_dict=True)
    overall_metrics = {
        "Precision": classification_result["weighted avg"]["precision"],
        "Recall": classification_result["weighted avg"]["recall"],
        "F1-Score": classification_result["weighted avg"]["f1-score"],
        "Accuracy": accuracy_score(y_test, y_pred_test),
    }

    # Display overall metrics in a table
    st.write("Overall Metrics (KNN):")
    df_metrics = pd.DataFrame(overall_metrics, index=[0])
    st.table(df_metrics)

    # Plot confusion matrix as a heatmap
    st.write("Confusion Matrix (KNN):")
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Display the plot using st.pyplot with the figure
    fig, ax = plt.subplots()
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    st.pyplot(fig)


st.title("Anomaly Detection")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader_1")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    algorithm = st.selectbox(
        "Select algorithm",
        ["LSTM", "Isolation Forest", "SVM", "KNN"],
    )

    if st.button("Submit"):
        if algorithm == "LSTM":
            X_data = df.drop(labels=["intrusion"], axis=1).values
            y_data = df["intrusion"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.20, random_state=42
            )

            lstm(X_train, y_train, X_test, y_test)
        elif algorithm == "Isolation Forest":
            isolation_forest(X_train, X_test)
        elif algorithm == "SVM":
            svm(X_train, X_test)
        elif algorithm == "KNN":
            knn(X_train, X_test)
