## Task_2 EDA
## Overview
The `Task_2_EDA.ipynb` notebook performs exploratory data analysis (EDA) on the loaded dataset. Key tasks include:

### Features
1. **Summary Statistics**:
   Provides insights into central tendency, dispersion, and distributions of numerical features.
2. **Visualization**:
   - Histograms for numerical features.
   - Count plots for categorical features.
   - Box plots for outlier detection.
3. **Correlation Analysis**:
   Examines relationships between numerical features.

### Observations
- Outliers detected in columns: `Amount`, `Value`, `PricingStrategy`, `FraudResult`.
- Significant skewness in numerical distributions.
- Imbalanced categories in features like `FraudResult` and `PricingStrategy`.

## Usage
1. Open the notebook:
   ```bash
   jupyter notebook notebooks/Task_2_EDA.ipynb
   ```
2. Follow the structured cells to review the dataset and outputs.
3. Modify or add additional analysis as needed for your requirements.

### Task 3 and Task 4 - Feature Engineering, RFMS, and WoE
1. **Feature Engineering**:
   - Aggregated features such as total transaction amount, average transaction amount, transaction count, and standard deviation.
   - Extracted temporal features like transaction hour, day, month, and year.
   - Encoded categorical features using One-Hot and Label Encoding.
   - Normalized and standardized numerical features.

2. **RFMS and WoE Binning**:
   - Constructed Recency, Frequency, and Monetary (RFMS) features.
   - Assigned good/bad labels using RFMS scores.
   - Performed Weight of Evidence (WoE) binning for interpretability and predictive modeling.

#### Usage for Task 3 and Task 4:
1. Open the notebook:
   ```bash
   jupyter notebook notebooks/Task_3_and_4.ipynb
   ```
2. Follow the cells to execute feature engineering, RFMS calculations, and WoE binning.
3. Review and save the processed dataset for modeling.

### Task 5 - Modeling
1. **Model Selection and Training**:
   - Split data into training and testing sets.
   - Trained multiple models: Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting Machines (GBM).
   - Performed hyperparameter tuning using Grid Search for model optimization.

2. **Model Evaluation**:
   - Assessed models using metrics like Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
   - Selected the best model based on performance metrics.

3. **Model Saving**:
   - Saved the best-performing model for deployment.

#### Usage for Task 5:
1. Open the notebook:
   ```bash
   jupyter notebook notebooks/Task_5_modelling.ipynb
   ```
2. Execute the cells to train, evaluate, and save models.
3. Utilize the saved model for integration into applications or APIs.
