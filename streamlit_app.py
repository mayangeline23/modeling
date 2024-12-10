# Sidebar for dataset selection and file uploader
st.sidebar.header("Dataset Options")
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset to explore:",
    ["Select", "Heart Disease", "Diabetes", "Breast Cancer", "Liver Disorders"]
)

uploaded_file = st.sidebar.file_uploader("Upload a custom dataset (CSV):", type=["csv"])

if uploaded_file:
    # Load and display the uploaded dataset
    st.subheader("Uploaded Dataset Overview")
    data = pd.read_csv(uploaded_file)
    st.write(f"Shape: {data.shape}")
    st.write(data.head())
    # Allow the user to select target column
    target_column = st.selectbox("Select the target column:", data.columns)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
else:
    # Load selected dataset if no file is uploaded
    if dataset_choice != "Select":
        st.subheader(f"You selected: {dataset_choice}")
        
        # Dataset information and loading logic (as in your existing code)
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

        # Display dataset overview
        st.subheader("Dataset Overview")
        st.write(f"Shape: {data.shape}")
        st.write(data.head())
    else:
        st.write("Please select a dataset from the sidebar or upload a custom dataset to begin.")

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
