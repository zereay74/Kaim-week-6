from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

class CreditScoringRFMS:
    def __init__(self, df):
        self.df = df

    def compute_rfms(self):
        """Calculate RFMS metrics."""
        # Recency: Days since the last transaction
        self.df['recency'] = (self.df['TransactionStartTime'].max() - self.df['TransactionStartTime']).dt.days
        
        # Frequency: Count of transactions per customer
        frequency = self.df.groupby('CustomerId')['TransactionId'].count().rename('frequency')
        
        # Monetary: Total transaction amount per customer
        monetary = self.df.groupby('CustomerId')['Amount'].sum().rename('monetary')
        
        # Combine RFMS metrics into a single DataFrame
        rfms = self.df[['CustomerId']].drop_duplicates().merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')
        rfms = rfms.merge(self.df[['CustomerId', 'recency']].groupby('CustomerId').min(), on='CustomerId')
        
        # Normalize RFMS scores
        rfms['recency_norm'] = 1 - (rfms['recency'] / rfms['recency'].max())  # Normalize so lower recency is better
        rfms['frequency_norm'] = rfms['frequency'] / rfms['frequency'].max()
        rfms['monetary_norm'] = rfms['monetary'] / rfms['monetary'].max()
        
        # Calculate RFMS Score (weighted sum example)
        rfms['rfms_score'] = (
            0.3 * rfms['recency_norm'] + 
            0.4 * rfms['frequency_norm'] + 
            0.3 * rfms['monetary_norm']
        )
        return rfms

    def assign_good_bad_labels(self, rfms):
        """Assign good/bad labels based on RFMS score threshold."""
        threshold = rfms['rfms_score'].mean()  # Example: Use mean score as the threshold
        rfms['good_bad_label'] = np.where(rfms['rfms_score'] >= threshold, 'Good', 'Bad')
        return rfms

    def woe_binning(self, rfms, target_col='good_bad_label', feature_col='rfms_score', bins=5):
        """Perform WoE binning."""
        # Discretize the feature into bins
        binner = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        rfms['binned'] = binner.fit_transform(rfms[[feature_col]]).astype(int)
        
        # Calculate WoE for each bin
        woe_data = rfms.groupby('binned').agg(
            good_count=(target_col, lambda x: (x == 'Good').sum()),
            bad_count=(target_col, lambda x: (x == 'Bad').sum())
        )
        woe_data['total_count'] = woe_data['good_count'] + woe_data['bad_count']
        woe_data['good_pct'] = woe_data['good_count'] / woe_data['good_count'].sum()
        woe_data['bad_pct'] = woe_data['bad_count'] / woe_data['bad_count'].sum()
        woe_data['woe'] = np.log(woe_data['good_pct'] / woe_data['bad_pct'])
        
        return woe_data, rfms

# Example Usage
'''
# Initialize the class with the feature-engineered data
rfms_model = CreditScoringRFMS(encoded_df)

# Step 1: Compute RFMS scores
rfms = rfms_model.compute_rfms()
print("RFMS Metrics:")
print(rfms)

# Step 2: Assign Good/Bad Labels
rfms_labeled = rfms_model.assign_good_bad_labels(rfms)
print("\nRFMS with Good/Bad Labels:")
print(rfms_labeled)

# Step 3: Perform WoE Binning
woe_result, rfms_binned = rfms_model.woe_binning(rfms_labeled, feature_col='rfms_score')
print("\nWoE Binning Results:")
print(woe_result)
'''