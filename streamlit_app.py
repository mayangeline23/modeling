# Correcting the column drop based on the dataset
if dataset_choice == "Heart Disease":
    data = load_heart_disease_data()
    X = data.drop("target", axis=1)  # Drop the target column 'target'
    y = data["target"]
elif dataset_choice == "Diabetes":
    data = load_diabetes_data()
    X = data.drop("target", axis=1)  # Drop the target column 'target'
    y = data["target"]
elif dataset_choice == "Breast Cancer":
    data = load_breast_cancer_data()
    X = data.drop("target", axis=1)  # Drop the target column 'target'
    y = data["target"]
elif dataset_choice == "Liver Disorders":
    data = load_liver_disorders_data()
    X = data.drop("selector", axis=1)  # Drop the 'selector' column
    y = data["selector"]
elif dataset_choice == "Parkinson's Disease":
    data = load_parkinsons_data()
    if "class" in data.columns:  # Ensure 'class' column exists
        X = data.drop("class", axis=1)  # Drop the 'class' column
        y = data["class"]
    else:
        st.error("'class' column not found in Parkinson's Disease dataset")
elif dataset_choice == "Hepatitis":
    data = load_hepatitis_data()
    if "class" in data.columns:  # Ensure 'class' column exists
        X = data.drop("class", axis=1)  # Drop the 'class' column
        y = data["class"]
    else:
        st.error("'class' column not found in Hepatitis dataset")
