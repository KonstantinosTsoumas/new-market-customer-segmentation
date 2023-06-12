# Import libraries
import pandas as pd
import config
import argparse
import os
import joblib
from outlier_operations import detect_remove_plot_outliers
from data_cleaning import clean_missing_values, visualize_missing_values
from data_encoding import data_encoding
from data_transformation import convert_to_integer, perform_binning
from create_folds import create_folds
from correlation_analysis import perform_correlation_analysis
from sklearn.model_selection import train_test_split
from hyperparameter_search import tune_hyperparameters
from model_dispatcher import get_model
from train import run
from inference import predict


if __name__ == "__main__":
    # Initialize ArgumentParser class of argparse
    # and take input from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument("--model", type=str)
    #parser.add_argument("--tune_hyperparameters", action="store_true")
    args = parser.parse_args()

    # Read the data
    df_original = pd.read_csv(config.DATASET)

    # Clean the missing values
    df_cleaned = clean_missing_values(df_original)

    # Visualize the changes in missing values
    visualize_missing_values(df_original, df_cleaned)

    # Define columns to check and remove outliers from
    columns_to_check = ['Age', 'Family_Size', 'Work_Experience']
    df_cleaned = detect_remove_plot_outliers(df_cleaned, columns_to_check)

    # Convert 'Work_Experience', 'Family_Size' to integer data type
    df_cleaned = convert_to_integer(df_cleaned, 'Work_Experience')
    df_cleaned = convert_to_integer(df_cleaned, 'Family_Size')

    # Define the bin ranges and labels
    age_bins = [18, 35, 50, 65, float('inf')]
    age_labels = ['18-34', '35-49', '50-64', '65+']
    work_bins = [0, 3, 6, 9, float('inf')]
    work_labels = ['0-2', '3-5', '6-8', '9+']

    # Perform binning on the 'Age' and 'Work_Experience' column
    df_cleaned = perform_binning(df_cleaned, 'Age', age_bins, age_labels)
    df_cleaned = perform_binning(df_cleaned, 'Work_Experience', work_bins, work_labels)

    # Define the mapping for binary encoding
    mapping = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}

    # Call the data_encoding function
    df_encoded = data_encoding("BinaryEncoding", df_cleaned, ['Gender', 'Ever_Married', 'Graduated'], mapping=mapping)

    # Perform one hot encoding on the specified columns
    encoding_columns = ['Age_Bin', 'Profession', 'Work_Experience_Bin', 'Spending_Score', 'Family_Size', 'Var_1']
    df_encoded = data_encoding("OneHotEncoding", df_encoded, encoding_columns)

    # Dropping 'ID' to avoid redundancy and reduce multicollinearity
    df_encoded.drop('ID', axis=1, inplace=True)

    # Check correlations between variables after encoding
    perform_correlation_analysis(df_encoded)

    # Perform a 70/30 train-to-test split
    train_data, test_data = train_test_split(df_encoded, test_size=0.3, random_state=42)
    train_data.to_csv(config.TRAINING_FILE, index=False)
    test_data.to_csv(config.TEST_FILE, index=False)

    # Create folds
    cv_data = create_folds(train_data, args.n_splits)
    print(cv_data['kfold'].unique())

    # Save the training data with folds to a new CSV file
    cv_data.to_csv(config.TRAINING_FOLDS_FILE, index=False)

    # Run hyperparameter tuning and get the best model
    best_hyperparameters = tune_hyperparameters(cv_data, args.model, get_model)
    best_model = get_model(args.model).set_params(**best_hyperparameters)

    # Save the best model
    model_path = os.path.join(config.MODELS_FOLDER, f"{args.model}_best_model.bin")
    joblib.dump(best_model, model_path)

    # Run the training and evaluation for each fold and model
    for fold in range(args.n_splits):
        run(fold=fold, model=args.model)

    # Make predictions on the test set using the best model
    predictions = predict(model_path, config.TEST_FILE)
