# Import libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import config

def clean_missing_values(df_original):
    """
    This function clean the missing values in the training dataset in a column by column approach.
    The reference point for the missing values is the exploration notebook.
    Args:
        df_cleaned: Dataframe, Training dataset.
    returns:
        df_cleaned: Dataframe, Cleaned training and test datasets.
    """
    try:
        df_cleaned = df_original.copy()  # Create a copy of the original DataFrame

        # Remove rows with missing values in more than 3 columns in the train dataset
        df_cleaned = df_cleaned.loc[df_cleaned.isnull().sum(axis=1) < 3].copy()

        # Impute categories based on underlying patterns
        df_cleaned.loc[(df_cleaned['Var_1'].isnull()) & (df_cleaned['Graduated'] == 'Yes'), 'Var_1'] = 'Cat_6'
        df_cleaned.loc[(df_cleaned['Var_1'].isnull()) & (df_cleaned['Graduated'] == 'No'), 'Var_1'] = 'Cat_4'
        df_cleaned.loc[(df_cleaned['Var_1'].isnull()) & (df_cleaned['Profession'].isin(['Lawyer', 'Artist'])), 'Var_1'] = 'Cat_6'
        df_cleaned.loc[(df_cleaned['Var_1'].isnull()) & (df_cleaned['Age'] > 40), 'Var_1'] = 'Cat_6'

        # Impute missing values of 'Ever_Married' column
        df_cleaned.loc[(df_cleaned['Ever_Married'].isnull()) & (
        df_cleaned['Spending_Score'].isin(['Average', 'High'])), 'Ever_Married'] = 'Yes'
        df_cleaned.loc[(df_cleaned['Ever_Married'].isnull()) & (df_cleaned['Spending_Score'] == 'Low'), 'Ever_Married'] = 'No'
        df_cleaned.loc[(df_cleaned['Ever_Married'].isnull()) & (df_cleaned['Age'] > 40), 'Ever_Married'] = 'Yes'
        df_cleaned.loc[(df_cleaned['Ever_Married'].isnull()) & (df_cleaned['Profession'] == 'Healthcare'), 'Ever_Married'] = 'No'

        # Impute missing values of 'Profession' column
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Work_Experience'] > 8), 'Profession'] = 'Homemaker'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Age'] > 70), 'Profession'] = 'Lawyer'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Family_Size'] < 3), 'Profession'] = 'Lawyer'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Spending_Score'] == 'Average'), 'Profession'] = 'Artist'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Graduated'] == 'Yes'), 'Profession'] = 'Artist'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Ever_Married'] == 'Yes'), 'Profession'] = 'Artist'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Ever_Married'] == 'No'), 'Profession'] = 'Healthcare'
        df_cleaned.loc[(df_cleaned['Profession'].isnull()) & (df_cleaned['Spending_Score'] == 'High'), 'Profession'] = 'Executives'

        # Impute missing values of the 'Work_Experience' column (with pattern and median imputation)
        df_cleaned['Work_Experience'].fillna(df_cleaned['Work_Experience'].median(), inplace=True)

        # Impute missing values in the 'Family_Size' column in a hybrid way (via pattern and mode imputation)
        df_cleaned.loc[(df_cleaned['Family_Size'].isnull()) & (df_cleaned['Ever_Married'] == 'Yes'), 'Family_Size'] = 2.0
        df_cleaned.loc[(df_cleaned['Family_Size'].isnull()) & (df_cleaned['Var_1'] == 'Cat_6'), 'Family_Size'] = 2.0
        df_cleaned.loc[(df_cleaned['Family_Size'].isnull()) & (df_cleaned['Graduated'] == 'Yes'), 'Family_Size'] = 2.0

        # Impute missing values with the mode
        df_cleaned['Family_Size'].fillna(df_cleaned['Family_Size'].mode()[0], inplace=True)

        # Impute missing values of the 'Graduated' column in a hybrid way (with pattern and mode imputation)
        df_cleaned.loc[(pd.isnull(df_cleaned["Graduated"])) & (df_cleaned['Spending_Score'] == 'Average'), "Graduated"] = 'Yes'
        df_cleaned.loc[(pd.isnull(df_cleaned["Graduated"])) & (df_cleaned['Profession'] == 'Artist'), "Graduated"] = 'Yes'
        df_cleaned.loc[(pd.isnull(df_cleaned["Graduated"])) & (df_cleaned['Age'] > 49), "Graduated"] = 'Yes'
        df_cleaned.loc[(pd.isnull(df_cleaned["Graduated"])) & (df_cleaned['Var_1'] == 'Cat_4'), "Graduated"] = 'No'
        df_cleaned.loc[(pd.isnull(df_cleaned["Graduated"])) & (df_cleaned['Ever_Married'] == 'Yes'), "Graduated"] = 'Yes'

        # Impute remaining missing values with the mode
        df_cleaned["Graduated"].fillna(df_cleaned["Graduated"].mode()[0], inplace=True)

        # Save the cleaned data as CSV
        output_path = os.path.join(config.INPUT_FOLDER, 'df_cleaned.csv')
        df_cleaned.to_csv(output_path, index=False)

    except Exception as e:
        print(f"An error occurred during data cleaning, please check again: {str(e)}")
        return None

    return df_cleaned

def visualize_missing_values(df_original, df_cleaned):
    """
    Visualizes the comparison of missing values before and after cleaning.
    Args:
        df_before: Dataframe, original dataset
        df_after: Dataframe, cleaned dataset
        fig_path: str, path to save the graph
    Returns: -
    """
    # Get the null values of original and cleaned datasets
    missing_values_before = df_original.isnull().sum()
    missing_values_after = df_cleaned.isnull().sum()

    # Create before and after plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(missing_values_before.index, missing_values_before.values)
    ax[0].set_title('Missing Values Before Cleaning')
    ax[0].set_xlabel('Columns')
    ax[0].set_ylabel('Count')
    ax[0].tick_params(axis='x', rotation=90)
    ax[1].bar(missing_values_after.index, missing_values_after.values)
    ax[1].set_title('Missing Values After Cleaning')
    ax[1].set_xlabel('Columns')
    ax[1].set_ylabel('Count')
    ax[1].tick_params(axis='x', rotation=90)

    # Adjust subplot parameters
    plt.tight_layout()

    # Save the plot
    fig_path = os.path.join(config.VISUALS_FOLDER, "data_before_and_after_cleaning.png")
    plt.savefig(fig_path)
    plt.close()


