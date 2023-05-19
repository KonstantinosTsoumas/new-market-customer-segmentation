# Import libraries
import pandas as pd
import config
from data_cleaning import clean_missing_values, visualize_missing_values
from data_encoding import data_encoding
from data_scaler import scale_variables

# Read the data
df_original = pd.read_csv(config.TRAINING_FILE)

# Clean the missing values
df_cleaned = clean_missing_values(df_original)

# Visualize the changes in missing values
visualize_missing_values(df_original, df_cleaned)

# Convert 'Work_Experience' to numerical data type
df_cleaned['Work_Experience'] = df_cleaned['Work_Experience'].astype('int')

# Define columns for scaling and encoding
scaling_columns = ['Age', 'Work_Experience', 'Family_Size']
encoding_columns = ['Age', 'Profession', 'Work_Experience', 'Spending_Score', 'Family_Size']

# Apply data scaling
df_scaled = scale_variables("MinMaxScaling", df_cleaned, scaling_columns)

# Apply data encoding
df_encoded = data_encoding("OneHotEncoding", df_cleaned, encoding_columns)

# Dropping 'ID' to avoid redundancy and reduce multicollinearity
df_encoded.drop('ID', axis=1, inplace=True)

