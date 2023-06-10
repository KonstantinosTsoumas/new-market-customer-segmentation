import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import config


def detect_remove_plot_outliers(df, columns, threshold=3.5):
    """
    This function executes the outlier detection, removal, and scatter plot functions in sequence.

    Parameters:
    df: dataframe containing the data.
    columns: List of column names to check for outliers and create scatter plots.
    threshold: Modified z-score threshold above which data points are considered outliers (default = 3.5).
    """
    outliers = detect_outliers_modified_zscore(df, columns, threshold)
    # Add the outlier columns to the cleaned DataFrame
    for col in columns:
        df[f'{col}_outlier'] = outliers[f'{col}_outlier']

    plot_outliers_bar(df, columns)
    df_cleaned = remove_outliers(df, columns, outliers)

    return df_cleaned
def detect_outliers_modified_zscore(df, columns, threshold=3.5):
    """
    This function detects outliers using modified z-score method for multiple columns in a DataFrame.

    Args:
    df: dataframe containing the columns to check for outliers.
    columns: list of column names to check for outliers.
    threshold: modified z-score threshold above which data points are considered outliers (default = 3.5)

    Returns: A dataframe with added boolean outlier columns for each specified column.
    """
    outliers_df = pd.DataFrame(index=df.index)

    for col in columns:
        column = df[col]
        median = column.median()
        mad = (column - median).abs().median()
        modified_z_scores = 0.6745 * (column - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        outliers_df[f'{col}_outlier'] = outliers

    return pd.concat([df, outliers_df], axis=1)


def remove_outliers(df, columns, outliers):
    """
    This function removes outliers from a DataFrame based on the specified columns and outlier boolean values.

    Parameters:
    df: dataframe containing the columns to check for outliers.
    columns: list of column names to check for outliers.
    outliers: dataframe with boolean outlier columns.

    Returns: A dataframe with outliers removed.
    """
    for col in columns:
        df = df.loc[~outliers[f'{col}_outlier']]

    return df

def plot_outliers_bar(df, columns):
    """
    This function creates bar plots to visualize outliers for specified columns in a dataframe.

    Parameters:
    df: dataframe containing the data to plot.
    columns: List of column names for the bar plots.
    """
    for col in columns:
        plt.figure(figsize=(8, 6))

        # Filter the dataframe to include only outliers for the current column
        outliers = df[df[f'{col}_outlier']]

        if not outliers.empty:
            # Calculate the count of outliers in each category
            outlier_counts = outliers[col].value_counts()

            # Plot the bar chart
            outlier_counts.plot(kind='bar', color='red')

            plt.xlabel(col)
            plt.ylabel('Outlier Count')
            plt.title(f'Bar Plot of Outliers by {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot as an image in the config.VISUALS_FOLDER directory
            output_path = os.path.join(config.VISUALS_FOLDER, f'{col}_outliers_bar.png')
            plt.savefig(output_path)
            plt.close()

