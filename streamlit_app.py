import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
        "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    data = pd.read_csv(url, header=None, names=columns)
    data.replace("?", pd.NA, inplace=True)
    return data.dropna().astype(float)

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
    
    # ROC curve and AUC calculation
    if len(y.unique()) > 2:
        # If target is not binary, map it to {0, 1}
        y_test = y_test.replace(2, 1)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return accuracy_score(y_test, y_pred), pd.DataFrame(report).transpose(), fpr, tpr, roc_auc

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
        "description": "This dataset contains medical records of patients used for predicting diabetes. The target variable indicates whether the patient has diabetes (1) or not (0).",
        "source": "https://archive.ics.uci.edu/ml/datasets/diabetes",
        "attributes": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
            "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
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

# Create a sidebar for dataset selection
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset to explore:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

# File upload section (separate from the dataset select box)
uploaded_file = st.sidebar.file_uploader("Your upload your CSV file", type="csv")

# Render only after a dataset or file is selected
if dataset_choice != "Select":
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

elif uploaded_file is not None:
    # Handle uploaded file
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(data.head())  # Preview of uploaded data
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # Last column as the target
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape}")
    st.write(data.head())

else:
    st.write("Please select a dataset from the sidebar to begin.")

# Model Selection and Evaluation
if 'data' in locals():
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["RandomForest", "LogisticRegression", "GradientBoosting"]
    )

    if st.button("Train and Evaluate Model"):
        st.subheader("Model Performance")
        accuracy, report_df, fpr, tpr, roc_auc = train_and_evaluate_model(X, y, model_type=model_choice)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.dataframe(report_df)

        # Feature Importance for Tree-based models
        if model_choice in ["RandomForest", "GradientBoosting"]:
            model = RandomForestClassifier(random_state=42) if model_choice == "RandomForest" else GradientBoostingClassifier(random_state=42)
            model.fit(X, y)
            feature_importances = model.feature_importances_
            st.subheader("Feature Importances")
            sns.barplot(x=X.columns, y=feature_importances)
            plt.xticks(rotation=90)
            st.pyplot()

        # Plot ROC curve
        st.subheader("ROC Curve")
        st.write(f"AUC: {roc_auc:.2f}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        st.pyplot()

    # Visualization Options
    if st.sidebar.checkbox("Show Pairplot"):
        st.subheader("Pairplot")
        sns.pairplot(data)
        st.pyplot()

    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot()
