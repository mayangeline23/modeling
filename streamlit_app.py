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
st.title("UCI Disease Prediction Dashboard")

# Dataset descriptions
dataset_info = {
    "Heart Disease": {
        "description": "This dataset contains 303 records and 14 attributes related to diagnosing heart disease. The target variable indicates the presence of heart disease (1: disease, 0: no disease).",
        "source": "https://archive.ics.uci.edu/ml/datasets/Heart+Disease",
        "attributes": [
            "age", "sex", "cp (chest pain type)", "trestbps (resting blood pressure)",
            "chol (serum cholesterol)", "fbs (fasting blood sugar)", "restecg (resting ECG)",
            "thalach (max heart rate achieved)", "exang (exercise-induced angina)",
            "oldpeak (ST depression)", "slope (slope of peak exercise ST segment)",
            "ca (number of vessels colored by fluoroscopy)", "thal (thalassemia)", "target"
        ]
    },
    "Diabetes": {
        "description": "The diabetes dataset from sklearn consists of 442 records with 10 attributes. It is used for regression but has been modified here for classification by binarizing the target variable.",
        "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        "attributes": list(load_diabetes_data().columns)
    },
    "Breast Cancer": {
        "description": "This dataset contains 569 records and 30 attributes related to diagnosing breast cancer. The target variable indicates whether the cancer is malignant or benign.",
        "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        "attributes": list(load_breast_cancer_data().columns)
    },
    "Liver Disorders": {
        "description": "The liver disorders dataset contains 345 records and 7 attributes related to the diagnosis of liver disorders.",
        "source": "https://archive.ics.uci.edu/ml/datasets/Liver+Disorders",
        "attributes": ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
    }
}

# Sidebar: Allow dataset selection or file upload
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset to explore:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

# Checkbox to allow dataset upload
upload_data = st.sidebar.checkbox("Upload your own dataset")

# Handle file upload if checked
if upload_data:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset")
        st.write(data.head())  # Display a preview of the uploaded dataset
        # Set X and y based on available columns (assumes the last column is the target)
        X = data.iloc[:, :-1]  # All columns except the last one
        y = data.iloc[:, -1]   # Last column as the target

# Render the selected dataset
elif dataset_choice != "Select":
    st.subheader(f"You selected: {dataset_choice}")
    
    # Display dataset information
    st.markdown("### Dataset Information")
    st.write(dataset_info[dataset_choice]["description"])
    st.markdown(f"**Source:** [Dataset Link]({dataset_info[dataset_choice]['source']})")
    st.markdown("**Attributes:**")
    st.write(dataset_info[dataset_choice]["attributes"])

    # Load the appropriate dataset
    if dataset_choice == "Heart Disease":
        data = load_heart_disease_data()
        X = data.drop("target", axis=1)
        y = data["target"]
    elif dataset_choice == "Diabetes":
        data = load_diabetes_data()
        X = data.drop("target", axis=1)
        y = data["target"]
    elif dataset_choice == "Breast Cancer":
        data = load_breast_cancer_data()
        X = data.drop("target", axis=1)
        y = data["target"]
    elif dataset_choice == "Liver Disorders":
        data = load_liver_disorders_data()
        X = data.drop("selector", axis=1)
        y = data["selector"]

    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape}")
    st.write(data.head())

# Model Selection and Evaluation
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

    # Feature Importance for Tree-based models
    if model_choice in ["RandomForest", "GradientBoosting"]:
        model = RandomForestClassifier(random_state=42) if model_choice == "RandomForest" else GradientBoostingClassifier(random_state=42)
        model.fit(X, y)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.subheader("Feature Importance")
        st.bar_chart(feature_importances)

# Visualization Options
if 'data' in locals():
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
    st.write("Please select or upload a dataset to begin.")
