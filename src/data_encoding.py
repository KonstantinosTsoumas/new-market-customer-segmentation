import pandas as pd
import os
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from src import config

def label_encoding(data, columns):
    """
    This function applies label encoding to the specified columns of the data.
    Args:
        data: DataFrame, the dataframe containing the data to be encoded.
        columns: list, the list of columns to be encoded.
    Returns:
        DataFrame: the encoded DataFrame.
    """

    encoder = LabelEncoder()
    for column in columns:
        data[column] = encoder.fit_transform(data[column])

    # Save the cleaned data as CSV
    output_path = os.path.join(config.INPUT_FOLDER, 'encoded_data.csv')
    data.to_csv(output_path, index=False)

    return data

def one_hot_encoding(data, columns):
    """
    This function applies one-hot encoding to the specified columns of the data.
    Args:
        data: DataFrame, the dataframe containing the data to be encoded.
        columns: list, the list of columns to be encoded.
    Returns:
        DataFrame: the encoded DataFrame.
    """

    # Save the cleaned data as CSV
    output_path = os.path.join(config.INPUT_FOLDER, 'encoded_data.csv')
    data.to_csv(output_path, index=False)

    return pd.get_dummies(data, columns=columns)

def target_encoding(data, columns, target_column):
    """
    This function applies target encoding to the specified columns of the data using the target column.
    Args:
        data: DataFrame, the dataframe containing the data to be encoded.
        columns: list, the ist of columns to be encoded.
        target_column: str, the target column to be used for target encoding.
    Returns:
        DataFrame: The encoded DataFrame.
    """

    target_encoder = ce.TargetEncoder(cols=columns)
    data[columns] = target_encoder.fit_transform(data[columns], data[target_column])

    # Save the cleaned data as CSV
    output_path = os.path.join(config.INPUT_FOLDER, 'encoded_data.csv')
    data.to_csv(output_path, index=False)

    return data

def data_encoding(encoding_strategy, data, columns, target_column=None):
    """
    This function applies the specified encoding strategy to the given columns of the data.
    Args:
        encoding_strategy: str, the encoding strategy to be applied. Supported options: 'LabelEncoding', 'OneHotEncoding', 'TargetEncoding'
        data: DataFrame, the dataframe containing the data to be encoded.
        columns: list, the list of columns to be encoded.
        target_column: str, the target column to be used for target encoding. Default is None.
    Returns:
        DataFrame: the encoded DataFrame.
    Raises:
        ValueError: If an unsupported encoding strategy is provided or if the target column does not exist.
        TypeError: If the input arguments are of incorrect types.
    """
    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The 'data' argument must be a pandas DataFrame.")
    if not isinstance(columns, list):
        raise TypeError("The 'columns' argument must be a list.")

    if target_column and target_column not in data.columns:
        raise ValueError("The target column does not exist in the data DataFrame.")

    if encoding_strategy == 'LabelEncoding':
        return label_encoding(data, columns)
    elif encoding_strategy == 'OneHotEncoding':
        return one_hot_encoding(data, columns)
    elif encoding_strategy == 'TargetEncoding':
        if not target_column:
            raise ValueError("Target column must be provided for Target Encoding.")
        return target_encoding(data, columns, target_column)
    else:
        raise ValueError("Unsupported encoding strategy. Available options: 'LabelEncoding', 'OneHotEncoding', 'TargetEncoding'")



