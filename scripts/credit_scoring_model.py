from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
import joblib


class CreditScoringModel:
    def __init__(self, df, target_col, feature_cols):
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.models = {}
        self.best_model = None

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def train_models(self, X_train, y_train):
        """Train multiple models."""
        # Logistic Regression
        self.models['LogisticRegression'] = LogisticRegression(max_iter=1000, random_state=42)

        # Decision Tree
        self.models['DecisionTree'] = DecisionTreeClassifier(random_state=42)

        # Random Forest
        self.models['RandomForest'] = RandomForestClassifier(random_state=42)

        # Gradient Boosting Machine (GBM)
        self.models['GBM'] = GradientBoostingClassifier(random_state=42)

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using Grid Search."""
        param_grid = {
            'RandomForest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
            },
            'GBM': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
            },
        }

        for model_name, grid in param_grid.items():
            if model_name in self.models:
                print(f"Tuning {model_name}...")
                grid_search = GridSearchCV(self.models[model_name], grid, scoring='roc_auc', cv=3)
                grid_search.fit(X_train, y_train)
                self.models[model_name] = grid_search.best_estimator_

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and select the best one."""
        metrics = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            metrics[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': roc_auc,
            }

        self.best_model = max(metrics, key=lambda m: metrics[m]['ROC-AUC'])
        print(f"Best Model: {self.best_model}")
        return metrics

    def save_model(self, filepath):
        """Save the best model to disk."""
        if self.best_model:
            joblib.dump(self.models[self.best_model], filepath)
            print(f"Best model saved to {filepath}")
        else:
            print("No model to save!")

# Example Usage
'''
# Define features and target
target = 'FraudResult'  # or 'good_bad_label' from Task 4
features = ['recency', 'frequency', 'monetary', 'Amount', 'Value', 'CountryCode']  # Example feature columns

# Initialize and execute the pipeline
scoring_model = CreditScoringModel(df=feature_engineered_df, target_col=target, feature_cols=features)

# Split data
X_train, X_test, y_train, y_test = scoring_model.split_data()

# Train models
scoring_model.train_models(X_train, y_train)

# Hyperparameter tuning
scoring_model.hyperparameter_tuning(X_train, y_train)

# Evaluate models
metrics = scoring_model.evaluate_models(X_test, y_test)
print("\nModel Performance Metrics:")
for model, metric in metrics.items():
    print(f"{model}: {metric}")

# Save the best model
scoring_model.save_model('best_credit_model.pkl')
''' 