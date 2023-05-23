import os
import seaborn as sns
import matplotlib.pyplot as plt
import config

def perform_correlation_analysis(df):
    """
    This function performs correlation analysis on a dataFrame.
    Args:
        df: Dataframe, the input dataframe containing numerical variables for correlation analysis.
    Returns: -
    """

    # Keep only numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    boolean_columns = df.select_dtypes(include=['bool']).columns
    df_filtered = df[numerical_columns.union(boolean_columns)]

    try:
        # Calculate the correlation matrix
        correlation_matrix = df_filtered.corr()

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(18, 14))
        sns.set(font_scale=1)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=1, fmt=".2f")
        plt.title('Correlation Matrix')

        # Save the cleaned data as CSV
        output_path = os.path.join(config.INPUT_FOLDER, 'corr_matrix_after_encoding.png')
        plt.savefig(output_path)

    except ValueError as e:
        if str(e) == "zero-size array to reduction operation fmin which has no identity":
            print("Insufficient data for correlation analysis.")
        else:
            print("An error occurred during correlation analysis:", str(e))
