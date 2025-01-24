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
