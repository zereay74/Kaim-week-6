## Project Overview
This project focuses on analyzing, preparing, and modeling a dataset for **credit scoring purposes**. Key tasks include:
1. Identifying and handling missing values.
2. Visualizing distributions of numerical and categorical features.
3. Detecting outliers and performing correlation analysis.
4. Engineering features and constructing RFMS scores with WoE binning.
5. Training and evaluating machine learning models for credit scoring.
6. Deploying the trained model using a REST API with Docker containerization for scalability.

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
|-- test/                 # Testing folder (if applicable)
|-- model_serving_api.py  # fast api app
|-- Dockerfile
|-- README.md             # Main project README
|-- .gitignore            # To ignore data and model outputs
|-- requirements.txt
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
- **Model Deployment**:
  Serves the trained model via a REST API, enabling real-time predictions, and supports containerization with Docker.

## Getting Started
1. Clone the repository.
2. Install required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute scripts or open notebooks for data analysis.

### Task Breakdown
1. **Task 2 - EDA**:
   Open  upload data (csv) and run:
   ```bash
   jupyter notebook notebooks/Task_2_EDA.ipynb
   ```
   Review summary statistics, visualize data, and analyze correlations.

2. **Task 3 and Task 4 - Feature Engineering, RFMS, and WoE**:
   Open  upload data (csv) and run:
   ```bash
   jupyter notebook notebooks/Task_3_and_4.ipynb
   ```
   Execute feature engineering, RFMS calculations, and WoE binning.

3. **Task 5 - Modeling**:
   Open  upload data (csv) and run:
   ```bash
   jupyter notebook notebooks/Task_5_modelling.ipynb
   ```
   Train models, evaluate their performance, and save the best-performing model for deployment.

4. **Task 6 - Deployment and Docker Containerization**:
   - **Model Serving API**:
     Use the `model_serving_api.py` script to serve the trained model.
     ```bash
     uvicorn scripts.model_serving_api:app --host 0.0.0.0 --port 8000
     ```
     Access the API at `http://localhost:8000/docs`. Use the `/predict` endpoint to upload a CSV file for predictions.

   - **Docker Containerization**:
     Build and run the API in a Docker container for scalability.
     ```bash
     docker build -t fastapi-ml-predictor .
     docker run -d -p 8000:8000 username/fastapi-ml-predictor  # run docker container
     ```
     The API will be accessible at `http://localhost:8000/docs` in a containerized environment.
