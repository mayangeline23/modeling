# Streamlit App
st.title("UCI Disease Prediction Dashboard")

# Sidebar: Allow dataset selection or file upload
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset or upload your own:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders", "Upload File"]
)

# Handle file upload if "Upload File" is selected
if dataset_choice == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset")
        st.write(data.head())  # Display a preview of the uploaded dataset
        # Set X and y based on available columns (assumes the last column is the target)
        X = data.iloc[:, :-1]  # All columns except the last one
        y = data.iloc[:, -1]   # Last column as the target
        
        # Model Selection and Evaluation after file upload
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

# Handle predefined dataset choice (before the file upload option)
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

    # Model Selection and Evaluation (under the dataset preview section)
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

# Visualization Options (optional, can be at the bottom)
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
