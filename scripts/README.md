## Overview
`data_load_clean_transform.py` is a Python script that handles:
1. **Loading Data**: Reads the dataset from the `data/` folder.
2. **Cleaning Data**: Checks for missing values and performs imputation.
3. **Transforming Data**: Handles outliers and prepares the dataset for further analysis.

## Features
- **Missing Value Check**:
  Outputs a summary of missing values, percentages, and column data types.
- **Outlier Detection**:
  Identifies and marks columns with significant outliers.
- **Data Transformation**:
  Ensures numerical and categorical data are prepared for downstream analysis.

## Usage
Run the script:
```bash
python scripts/data_load_clean_transform.py
```
Modify the configuration as needed to adjust cleaning or transformation parameters