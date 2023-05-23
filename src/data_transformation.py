import pandas as pd

def convert_to_integer(data, column):
    """
    This function converts a specified column of a dataframe to integer type.

    Parameters:
        data: dataframe the input dataframe.
        column: str, the name of the column to convert to integer.

    Returns:
        pandas.DataFrame: the modified dataframe with the converted column.

    """
    data[column] = data[column].astype(int)
    return data

def perform_binning(data, column, bins, labels):
    """
    This function performs binning on a specified column of a dataframe.

    Args:
        data: dataframe, the input DataFrame.
        column: str, the name of the column to perform binning on.
        bins: list, the bin ranges for binning.
        labels: list, the bin labels for the resulting bins.

    Returns:
        Dataframe: the modified DataFrame with the binned column.

    """
    # Perform binning on the specified column
    data[column + '_Bin'] = pd.cut(data[column], bins=bins, labels=labels, right=True)
    # Drop the original column
    data.drop(columns=column, inplace=True)
    return data
