## Overview
`data_load_clean_transform.py` is a Python script that handles:
1. **Loading Data**: Reads the dataset from the `data/` folder.
2. **Cleaning Data**: Checks for missing values and performs imputation.
3. **Transforming Data**: Handles outliers and prepares the dataset for further analysis.

`feature_engineering.py` is a Python script designed for:
1. **Feature Engineering**: Aggregating features, extracting temporal features, encoding categorical variables, and scaling numerical features.

`credit_scoring_rfms.py` focuses on:
1. **RFMS Feature Construction**: Creating Recency, Frequency, and Monetary features.
2. **Good/Bad Label Assignment**: Classifying customers based on RFMS scores.
3. **Weight of Evidence (WoE) Binning**: Transforming features for interpretability and predictive modeling.

`credit_scoring_model.py` is tailored for:
1. **Model Selection and Training**: Implements machine learning models (Logistic Regression, Decision Trees, Random Forest, GBM) and hyperparameter tuning.
2. **Model Evaluation**: Provides metrics like Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
3. **Model Saving**: Saves the trained model for deployment.

## Features
- **Missing Value Check**:
  Outputs a summary of missing values, percentages, and column data types.
- **Outlier Detection**:
  Identifies and marks columns with significant outliers.
- **Data Transformation**:
  Ensures numerical and categorical data are prepared for downstream analysis.
- **Feature Engineering**:
  Includes feature aggregation, temporal feature extraction, encoding, and scaling.
- **RFMS and WoE**:
  Constructs RFMS features, assigns good/bad labels, and performs WoE binning.
- **Model Training**:
  Supports multiple ML models, hyperparameter tuning, model saving, and evaluation metrics.
## Usage
Run the notebooks to import and implement the scripts: 