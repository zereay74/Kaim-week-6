## Project Overview
This project focuses on analyzing, preparing, and modeling a dataset for credit scoring purposes. Key tasks include:
1. Identifying and handling missing values.
2. Visualizing distributions of numerical and categorical features.
3. Detecting outliers and performing correlation analysis.
4. Engineering features and constructing RFMS scores with WoE binning.
5. Training and evaluating machine learning models for credit scoring.

## Repository Structure
```
|-- data/                 # Dataset folder
|-- scripts/              # Python scripts for data processing
|   |-- data_load_clean_transform.py
|   |-- feature_engineering.py
|   |-- credit_scoring_rfms.py
|   |-- credit_scoring_model.py
|-- notebooks/            # Jupyter notebooks for tasks
|   |-- Task_2_EDA.ipynb
|   |-- Task_3_and_4.ipynb
|   |-- Task_5_modelling.ipynb
|-- test/              
|-- README.md             # Main project README
|-- .gitignore            # To ignore data and model outputs
```

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

## Getting Started
1. Clone the repository.
2. Install required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute scripts or open notebooks for data analysis.

### Task Breakdown
1. **Task 2 - EDA**:
   Open and run:
   ```bash
   jupyter notebook notebooks/Task_2_EDA.ipynb
   ```
   Review summary statistics, visualize data, and analyze correlations.

2. **Task 3 and Task 4 - Feature Engineering, RFMS, and WoE**:
   Open and run:
   ```bash
   jupyter notebook notebooks/Task_3_and_4.ipynb
   ```
   Execute feature engineering, RFMS calculations, and WoE binning.

3. **Task 5 - Modeling**:
   Open and run:
   ```bash
   jupyter notebook notebooks/Task_5_modelling.ipynb
   ```
   Train models, evaluate their performance, and save the best-performing model for deployment.
