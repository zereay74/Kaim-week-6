import os 
import sys
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to handle loading data from CSV files.
    """
    
    def __init__(self, file_path):
        """
        Initializes the DataLoader with the file path to the CSV file.

        :param file_path: str, path to the CSV file
        """
        self.file_path = file_path

    def load_csv(self):
        """
        Loads the CSV file into a pandas DataFrame.

        :return: pd.DataFrame containing the data from the CSV file
        """
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"Data successfully loaded from {self.file_path}")
            logger.info(f"DataFrame Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"Error: No data in file at {self.file_path}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading the file: {e}")
            return None
# Example usage:
# loader = DataLoader("path_to_your_file.csv")
# df = loader.load_data()



logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, dataframe):
        """
        Initialize the DataCleaner with a DataFrame.
        :param dataframe: pandas DataFrame to be cleaned.
        """
        self.df = dataframe

    def check_missing_values(self):
        """
        Check for missing values in each column.
        :return: DataFrame with columns, missing value count, and percentage.
        """
        logger.info("Checking for missing values in the DataFrame.")
        missing_info = self.df.isnull().sum()
        missing_percentage = (missing_info / len(self.df)) * 100
        logger.info("Missing values check completed.")
        return pd.DataFrame({
            'Column': self.df.columns,
            'Missing Values': missing_info,
            'Missing Percentage': missing_percentage,
            'Data Type': self.df.dtypes
        }).reset_index(drop=True)

    def remove_outliers(self, threshold=3):
        """
        Remove outliers from numerical columns based on Z-score.
        :param threshold: Z-score threshold for identifying outliers.
        """
        logger.info(f"Removing outliers with Z-score threshold of {threshold}.")
        try:
            numeric_cols = self.df.select_dtypes(include=['number'])
            z_scores = numeric_cols.apply(zscore)
            self.df = self.df[(np.abs(z_scores) < threshold).all(axis=1)]
            logger.info("Outlier removal completed.")
        except Exception as e:
            logger.error(f"An error occurred during outlier removal: {e}")
    ''' 
    def detect_outliers_plot(self, column):
        """
        Plot a boxplot to detect outliers in a numerical column.
        :param column: Name of the numerical column to plot.
        """
        logger.info(f"Generating outlier detection plot for column: {column}.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        if not np.issubdtype(self.df[column].dtype, np.number):
            logger.error(f"Column '{column}' is not numerical.")
            return
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[column])
            plt.title(f"Outlier Detection in {column}")
            plt.show()
            logger.info(f"Outlier detection plot for '{column}' generated successfully.")
        except Exception as e:
            logger.error(f"An error occurred while plotting outliers: {e}")
    
    ''' 
    def detect_outliers_plot(self, columns):
        """
        Plot boxplots to detect outliers in multiple numerical columns.

        Args:
            columns: A list of column names to analyze for outliers.
        """
        if not all(np.issubdtype(self.df[col].dtype, np.number) for col in columns):
            logger.error("One or more columns are not numerical.")
            return

        try:
            # Dynamically adjust the figure size based on the number of columns
            num_columns = len(columns)
            fig, axes = plt.subplots(
                nrows=num_columns, ncols=1, figsize=(12, 5 * num_columns), squeeze=False
            )

            for i, col in enumerate(columns):
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Identify outliers
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                non_outliers = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

                # Prepare data for boxplot-style visualization
                sns.boxplot(
                    ax=axes[i, 0], x=self.df[col], color="lightgray", showmeans=False, width=0.5
                )
                sns.stripplot(
                    ax=axes[i, 0], x=non_outliers[col], color="blue", alpha=0.7, jitter=True, size=5, label="Normal Data"
                )

                # Highlight outliers
                if not outliers.empty:
                    sns.stripplot(
                        ax=axes[i, 0], x=outliers[col], color="red", alpha=0.9, jitter=True, size=5, label="Outliers"
                    )

                # Annotate plot
                axes[i, 0].axvline(x=lower_bound, color="orange", linestyle="--", label="Lower Bound")
                axes[i, 0].axvline(x=upper_bound, color="purple", linestyle="--", label="Upper Bound")
                axes[i, 0].axvline(x=q1, color="green", linestyle="-.", label="Q1")
                axes[i, 0].axvline(x=q3, color="green", linestyle="-.", label="Q3")
                axes[i, 0].set_title(f"Outlier Detection in {col}", fontsize=14)

                # Add legend
                axes[i, 0].legend(loc="upper right")

            plt.tight_layout()
            plt.show()
            logger.info(f"Outlier detection plots for columns {', '.join(columns)} generated successfully.")

        except Exception as e:
            logger.error(f"An error occurred while plotting outliers: {e}")


    def transform_datetime(self, column, timezone):
        """
        Convert a datetime column to a specified timezone.
        :param column: Name of the datetime column.
        :param timezone: Target timezone (e.g., 'UTC', 'America/New_York').
        """
        logger.info(f"Transforming datetime column '{column}' to timezone '{timezone}'.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        try:
            self.df[column] = pd.to_datetime(self.df[column])
            self.df[column] = self.df[column].dt.tz_localize(None).dt.tz_localize(timezone)
            logger.info(f"Datetime transformation for column '{column}' completed.")
        except Exception as e:
            logger.error(f"An error occurred while transforming datetime: {e}")

    def fill_missing_values(self, strategy='mean', columns=None):
        """
        Fill missing values in specified columns.
        :param strategy: Strategy to fill missing values ('mean', 'median', 'mode').
        :param columns: List of columns to fill missing values. If None, all columns are processed.
        """
        logger.info(f"Filling missing values using strategy '{strategy}'.")
        if columns is None:
            columns = self.df.columns

        for column in columns:
            try:
                if self.df[column].dtype == 'object':
                    self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                else:
                    if strategy == 'mean':
                        self.df[column].fillna(self.df[column].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[column].fillna(self.df[column].median(), inplace=True)
                    elif strategy == 'mode':
                        self.df[column].fillna(self.df[column].mode()[0], inplace=True)
                logger.info(f"Missing values in column '{column}' filled successfully.")
            except Exception as e:
                logger.error(f"An error occurred while filling missing values for column '{column}': {e}")

    def drop_column(self, column):
        """
        Drop a specified column from the DataFrame.
        :param column: Name of the column to drop.
        """
        logger.info(f"Dropping column: {column}.")
        if column not in self.df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            return
        self.df.drop(columns=[column], inplace=True)
        logger.info(f"Column '{column}' dropped successfully.")

    def standardize_column_names(self):
        """
        Standardize column names by converting them to lowercase and replacing spaces with underscores.
        """
        logger.info("Standardizing column names.")
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        logger.info("Column names standardized successfully.")

    def remove_duplicates(self):
        """
        Remove duplicate rows from the DataFrame.
        """
        logger.info("Removing duplicate rows from the DataFrame.")
        self.df.drop_duplicates(inplace=True)
        logger.info("Duplicate rows removed successfully.")

    def remove_nulls_from_columns(self, columns):
        """
        Remove rows with null values in the specified columns from the DataFrame.
        :param columns: List of column names to check for null values.
        """
        logger.info(f"Removing rows with null values in columns: {columns}.")
        if not isinstance(columns, list):
            logger.error("Columns parameter should be a list of column names.")
            return
        
        missing_columns = [col for col in columns if col not in self.df.columns]
        if missing_columns:
            logger.error(f"The following columns do not exist in the DataFrame: {', '.join(missing_columns)}")
            return

        self.df.dropna(subset=columns, inplace=True)
        logger.info(f"Rows with null values in columns {columns} removed successfully.")

    def get_cleaned_data(self):
        """
        Retrieve the cleaned DataFrame.
        :return: Cleaned DataFrame.
        """
        logger.info("Retrieving the cleaned DataFrame.")
        return self.df
    def summary_statistics(self):
        """
        Provide summary statistics for numerical columns.
        :return: DataFrame with summary statistics.
        """
        logger.info("Calculating summary statistics.")
        return self.df.describe()

    def distribution_numerical_features(self):
        """
        Visualize the distribution of numerical features with an adaptive y-axis.
        """
        logger.info("Visualizing distribution of numerical features.")
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(8, 4))
            
            # Handle edge case where all values in the column are the same
            data_range = self.df[col].max() - self.df[col].min()
            if data_range == 0:
                logger.warning(f"Column {col} has constant values. Skipping visualization.")
                plt.text(0.5, 0.5, f"No variation in {col}", horizontalalignment='center', verticalalignment='center', fontsize=12)
                plt.axis('off')
                plt.show()
                continue

            # Calculate bin width dynamically
            bin_width = data_range / 30  # Adjust number of bins if needed
            bins = np.arange(self.df[col].min(), self.df[col].max() + bin_width, bin_width)
            
            # Plot the histogram
            plot = sns.histplot(self.df[col], bins=bins, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            # Dynamically adjust y-axis based on visible data
            y_max = max([bar.get_height() for bar in plot.patches])
            if y_max > 0:
                plt.ylim(0, y_max * 1.1)  # Add 10% padding to the y-axis

            plt.show()

    def distribution_categorical_features(self):
        """
        Visualize the distribution of categorical features with adaptive y-axis.
        """
        logger.info("Visualizing distribution of categorical features.")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            
            # Plot the distribution using countplot
            plot = sns.countplot(data=self.df, x=col)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Dynamically adjust y-axis based on data values
            y_max = max([bar.get_height() for bar in plot.patches]) if plot.patches else 0
            if y_max > 0:
                plt.ylim(0, y_max * 1.1)  # Add 10% padding to the maximum height
            else:
                logger.warning(f"Column {col} has no data to plot.")
                plt.text(0.5, 0.5, f"No data in {col}", horizontalalignment='center', verticalalignment='center', fontsize=12)
                plt.axis('off')

            plt.show()

    def correlation_analysis(self):
        """
        Analyze correlation between numerical features.
        :return: Correlation matrix.
        """
        logger.info("Performing correlation analysis.")
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numerical_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        return corr_matrix

 
'''

    # Initialize the cleaner
    cleaner = DataCleaner(df)

    # Check missing values
    print("Missing Values:")
    print(cleaner.check_missing_values())

    # Fill missing values
    cleaner.fill_missing_values()

    # Remove duplicates
    cleaner.remove_duplicates()

    # Standardize column names
    cleaner.standardize_column_names()

    # Transform datetime column
    cleaner.transform_datetime('joining_date', 'UTC')

    # Detect outliers plot
    print("\nOutlier Detection Plot:")
    cleaner.detect_outliers_plot('salary')

    # Remove outliers
    cleaner.remove_outliers()

    # Drop a column
    cleaner.drop_column('name')

    # Get cleaned data
    cleaned_data = cleaner.get_cleaned_data()
    print("\nCleaned DataFrame:")
    print(cleaned_data)
'''