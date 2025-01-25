import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

class FeatureEngineering:
    def __init__(self, df):
        self.df = df

    def create_aggregate_features(self):
        """Create aggregate features per customer."""
        aggregated = self.df.groupby('CustomerId').agg(
            total_transaction_amount=('Amount', 'sum'),
            average_transaction_amount=('Amount', 'mean'),
            transaction_count=('TransactionId', 'count'),
            transaction_amount_std=('Amount', 'std')
        ).reset_index()
        return aggregated

    def extract_date_features(self):
        """Extract date and time-related features."""
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['transaction_hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['transaction_day'] = self.df['TransactionStartTime'].dt.day
        self.df['transaction_month'] = self.df['TransactionStartTime'].dt.month
        self.df['transaction_year'] = self.df['TransactionStartTime'].dt.year
        return self.df

    def encode_categorical_variables(self, method='onehot'):
        """Encode categorical variables using One-Hot Encoding or Label Encoding."""
        categorical_columns = ['BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode', 
                               'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
        if method == 'onehot':
            onehot_encoder = OneHotEncoder()
            encoded_df = pd.DataFrame(onehot_encoder.fit_transform(self.df[categorical_columns]).toarray(),
                                      columns=onehot_encoder.get_feature_names_out(categorical_columns))
            self.df = pd.concat([self.df, encoded_df], axis=1).drop(columns=categorical_columns)
        elif method == 'label':
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                self.df[col] = label_encoder.fit_transform(self.df[col])
        return self.df

    def scale_numerical_features(self, method='standardize'):
        """Normalize or standardize numerical features."""
        numerical_columns = ['Amount', 'Value']
        if method == 'normalize':
            scaler = MinMaxScaler()
        elif method == 'standardize':
            scaler = StandardScaler()
        else:
            raise ValueError("Method must be either 'normalize' or 'standardize'")

        self.df[numerical_columns] = scaler.fit_transform(self.df[numerical_columns])
        return self.df
# Example Usage
'''
df = pd.DataFrame(data)

# Initialize the FeatureEngineering class
feature_engineer = FeatureEngineering(df)

# Aggregate Features
aggregate_features = feature_engineer.create_aggregate_features()
print("Aggregate Features:")
print(aggregate_features)

# Extract Date Features
df_with_date_features = feature_engineer.extract_date_features()
print("\nDataFrame with Date Features:")
print(df_with_date_features.head())

# Encode Categorical Variables
encoded_df = feature_engineer.encode_categorical_variables(method='onehot')
print("\nDataFrame with Encoded Categorical Variables:")
print(encoded_df.head())

# Scale Numerical Features
scaled_df = feature_engineer.scale_numerical_features(method='standardize')
print("\nDataFrame with Scaled Numerical Features:")
print(scaled_df.head())
''' 