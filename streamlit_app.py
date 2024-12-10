import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to train and evaluate models
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
st.title("UCI Disease Prediction Dashboard with File Upload")

# Sidebar for dataset selection or file upload
st.sidebar.title("Options")
file_upload = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
dataset_choice = st.sidebar.selectbox(
    "Or choose a preloaded dataset:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

# Handle uploaded file
if file_upload:
    st.subheader("Uploaded Dataset")
    try:
        uploaded_data = pd.read_csv(file_upload)
        st.write(f"Shape: {uploaded_data.shape}")
        st.dataframe(uploaded_data.head())
        
        # Let user choose target column
        target_column = st.sidebar.selectbox("Select the target column:", uploaded_data.columns)
        if target_column:
            X = uploaded_data.drop(target_column, axis=1)
            y = uploaded_data[target_column]

            # Model selection and training
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

                # Feature importance for tree-based models
                if model_choice in ["RandomForest", "GradientBoosting"]:
                    model = RandomForestClassifier(random_state=42) if model_choice == "RandomForest" else GradientBoostingClassifier(random_state=42)
                    model.fit(X, y)
                    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.subheader("Feature Importance")
                    st.bar_chart(feature_importances)
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

# Handle preloaded datasets
elif dataset_choice != "Select":
    # Load preloaded datasets here (similar to the earlier code)
    st.subheader(f"Preloaded Dataset: {dataset_choice}")
    # The logic for preloaded datasets goes here (reuse your earlier code)

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

else:
    st.write("Please upload a file or select a preloaded dataset to begin.")
