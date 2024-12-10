import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load datasets
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

# Unified function for training and evaluation
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
    return accuracy_score(y_test, y_pred), pd.DataFrame(report).transpose()

# Streamlit App
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset to explore:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

# File upload section
st.sidebar.markdown("### File Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv", help="Upload your custom dataset")

# Main content based on dataset or uploaded file
if dataset_choice != "Select" or uploaded_file is not None:
    st.title("Disease Prediction Dashboard")
    
    # Handle dataset selection or file upload
    if dataset_choice != "Select":
        # Load predefined dataset
        if dataset_choice == "Heart Disease":
            data = load_heart_disease_data()
            target_column = "target"
        elif dataset_choice == "Diabetes":
            data = load_diabetes_data()
            target_column = "target"
        elif dataset_choice == "Breast Cancer":
            data = load_breast_cancer_data()
            target_column = "target"
        elif dataset_choice == "Liver Disorders":
            data = load_liver_disorders_data()
            target_column = "selector"
        st.subheader(f"Dataset Overview: {dataset_choice}")
    elif uploaded_file is not None:
        # Handle uploaded file
        data = pd.read_csv(uploaded_file)
        target_column = st.sidebar.selectbox(
            "Select Target Column:", data.columns, help="Choose the column to use as the target (dependent variable)."
        )
        st.subheader("Uploaded Dataset Overview")
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Display data overview
    st.write(f"Shape: {data.shape}")
    st.write(data.head())

    # Model Selection and Evaluation
    st.sidebar.markdown("### Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["RandomForest", "LogisticRegression", "GradientBoosting"]
    )
    if st.button("Train and Evaluate Model"):
        st.subheader("Model Performance")
        accuracy, report_df = train_and_evaluate_model(X, y, model_type=model_choice)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(report_df)

        # Feature Importance for tree-based models
        if model_choice in ["RandomForest", "GradientBoosting"]:
            model = RandomForestClassifier(random_state=42) if model_choice == "RandomForest" else GradientBoostingClassifier(random_state=42)
            model.fit(X, y)
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("Feature Importance")
            st.bar_chart(feature_importances)
    
    # Visualization Options
    st.sidebar.markdown("### Visualization Options")
    if st.sidebar.checkbox("Show Pairplot"):
        st.subheader("Pairplot")
        sns.pairplot(data)
        st.pyplot()

    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot()

else:
    st.title("UCI Disease Prediction Dashboard")
    st.write("Please select a dataset from the sidebar or upload your CSV file to begin.")
