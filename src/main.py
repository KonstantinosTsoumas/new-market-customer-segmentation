# Import libraries
import pandas as pd
import config
import argparse
from data_cleaning import clean_missing_values, visualize_missing_values
from data_encoding import data_encoding
from data_scaler import scale_variables
from data_transformation import convert_to_integer, perform_binning
from create_folds import create_folds

# Initialize ArgumentParser class of argparse
# and take input from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--n_splits', type=int, default=5)
args = parser.parse_args()

# Read the data
df_original = pd.read_csv(config.TRAINING_FILE)

# Clean the missing values
df_cleaned = clean_missing_values(df_original)

# Visualize the changes in missing values
visualize_missing_values(df_original, df_cleaned)

# Convert 'Work_Experience', 'Family_Size' to integer data type
df_cleaned = convert_to_integer(df_cleaned, 'Work_Experience')
df_cleaned = convert_to_integer(df_cleaned, 'Family_Size')

# Define the bin ranges and labels
age_bins = [18, 35, 50, 65, float('inf')]
age_labels = ['18-34', '35-49', '50-64', '65+']
work_bins = [0, 3, 6, 9, float('inf')]
work_labels = ['0-2', '3-5', '6-8', '9+']

# Perform binning on the 'Age' column
df_cleaned = perform_binning(df_cleaned, 'Age', age_bins, age_labels)
df_cleaned = perform_binning(df_cleaned, 'Work_Experience', work_bins, work_labels)

# Define columns for encoding
encoding_columns = ['Age_Bin', 'Profession', 'Work_Experience_Bin', 'Spending_Score', 'Family_Size']

# Apply data encoding
df_encoded = data_encoding("OneHotEncoding", df_cleaned, encoding_columns)

# Dropping 'ID' to avoid redundancy and reduce multicollinearity
df_encoded.drop('ID', axis=1, inplace=True)

# Create folds
df_train_folds = create_folds(df_encoded, args.n_splits)

# Save the training data with folds to a new CSV file
df_train_folds.to_csv(config.TRAINING_FOLDS_FILE, index=False)
