import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Helper functions
@st.cache_data
def load_heart_disease_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", 
        "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, header=None, names=columns)
    data.replace("?", pd.NA, inplace=True)
    return data.dropna().astype(float)

@st.cache_data
def load_diabetes_data():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    data = diabetes.data
    data["target"] = (diabetes.target > 140).astype(int)  # Binarize target
    return data

@st.cache_data
def load_breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer(as_frame=True)
    data = cancer.data
    data["target"] = cancer.target
    return data

@st.cache_data
def load_liver_disorders_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
    columns = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
    data = pd.read_csv(url, header=None, names=columns)
    return data

def train_and_evaluate_model(X, y, model_type="RandomForest"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy_score(y_test, y_pred), pd.DataFrame(report).transpose(), model

# Load datasets dynamically
def load_dataset(dataset_choice):
    if dataset_choice == "Heart Disease":
        data = load_heart_disease_data()
        target_col = "target"
    elif dataset_choice == "Diabetes":
        data = load_diabetes_data()
        target_col = "target"
    elif dataset_choice == "Breast Cancer":
        data = load_breast_cancer_data()
        target_col = "target"
    elif dataset_choice == "Liver Disorders":
        data = load_liver_disorders_data()
        target_col = "selector"
    else:
        return None, None, None
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return data, X, y

# Streamlit App Layout
st.title("Healthcare Dataset Explorer & Model Trainer")

# Sidebar - Dataset Selection
st.sidebar.header("Dataset Options")
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

# Sidebar - Model Upload
st.sidebar.header("Model Implementation")
uploaded_model = st.sidebar.file_uploader("Upload Pre-trained Model (.pkl)", type="pkl")

if dataset_choice != "Select" or uploaded_file is not None:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        data, X, y = load_dataset(dataset_choice)

    if data is not None:
        st.subheader("Dataset Overview")
        st.write(f"Shape: {data.shape}")
        st.dataframe(data.head())

        # Model Selection and Training
        model_choice = st.sidebar.selectbox(
            "Choose a model:",
            ["RandomForest", "LogisticRegression", "GradientBoosting"]
        )

        if st.button("Train and Evaluate Model"):
            st.subheader("Model Performance")
            accuracy, report_df, trained_model = train_and_evaluate_model(X, y, model_type=model_choice)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.dataframe(report_df)

            if model_choice in ["RandomForest", "GradientBoosting"]:
                st.subheader("Feature Importance")
                feature_importances = pd.Series(trained_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                st.bar_chart(feature_importances)

        # Visualizations
        if st.sidebar.checkbox("Show Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 6))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot()

if uploaded_model is not None:
    st.sidebar.success("Model file uploaded successfully!")
    model = pickle.load(uploaded_model)
    st.sidebar.write("Ready to use uploaded model for inference!")
