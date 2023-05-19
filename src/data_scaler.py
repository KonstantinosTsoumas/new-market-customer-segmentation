import pandas as pd
import config
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scale_variables(scaling_method, data, columns):
    """
    This function scales the variables in the given columns using the specified scaling method.

    Args:
        scaling_method: str, the scaling method to be applied (options: 'MinMaxScaling', 'StandardScaling').
        data: Dataframe, the dataframe containing the data to be scaled.
        columns: list, the list of columns to be scaled.

    Returns:
        DataFrame: the scaled dataframe.

    Raises:
        ValueError: If an unsupported scaling method is provided or if the columns do not exist in the data.
        TypeError: If the input arguments are of incorrect types.
    """

    # Validate input types
    if not isinstance(scaling_method, str):
        raise TypeError("The 'scaling_method' argument must be a string.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The 'data' argument must be a pandas DataFrame.")
    if not isinstance(columns, list):
        raise TypeError("The 'columns' argument must be a list.")

    # Check if the columns exist in the data
    if any(col not in data.columns for col in columns):
        raise ValueError("One or more of the specified columns do not exist in the data DataFrame.")

    try:
        if scaling_method == "MinMaxScaling":
            scaler = MinMaxScaler()
        elif scaling_method == "StandardScaling":
            scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaling method. Available options: 'MinMaxScaling', 'StandardScaling'.")

        # Scale the specified columns
        data[columns] = scaler.fit_transform(data[columns])
    except ValueError as e:
        raise ValueError("Error occurred while scaling variables: " + str(e))

    # Save the cleaned data as CSV
    output_path = os.path.join(config.INPUT_FOLDER, 'scaled_data.csv')
    data.to_csv(output_path, index=False)

    return data