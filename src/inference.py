import pandas as pd
import joblib
import argparse
import config
import os
from visualize_models import visualize_functions, plot_classification_report

def predict(model_file, test_data):
    """
    This function loads a trained model from a file and makes predictions on the test set.
    Args:
        model_file: str, the path to the trained model file.
        test_data: DataFrame, the test dataset.
    Returns:
        DataFrame: Predicted values for the test dataset.
    """
    # Load the trained model
    model = joblib.load(model_file)

    # Load the test data
    df_test = pd.read_csv(config.TEST_FILE)

    # Extract the features from the test dataset
    x_test = df_test.drop("Segmentation", axis=1).values
    y_test = df_test["Segmentation"].values

    # Fit the model on the test features
    fit = model.fit(x_test, y_test)

    # Make predictions on the test set
    preds = model.predict(x_test)

    # Create a DataFrame to store the predictions
    results_df = pd.DataFrame({"Segmentation": preds})

    # Specify the file path to save the figure
    figure_path = os.path.join(config.VISUALS_FOLDER, f"test_{model}.png")

    # Create the plot
    fig, ax = plot_classification_report(y_test, preds,
                                         title=f'test_dataset_report_{model}',
                                         figsize=(8, 6), dpi=70,
                                         target_names=["A", "B", "C", "D"])

    # visualize the model if specified in the visualize_functions dictionary
    if model in visualize_functions and visualize_functions[model] is not None:
            visualize_functions[model](fit, x_test, y_test)


    return results_df

if __name__ == "__main__":
    # Initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # Add model file argument
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Path to the trained model file."
    )

    # Read the arguments from the command line
    args = parser.parse_args()

    # Load the test dataset
    df_test = pd.read_csv(config.TEST_FILE)

    # Make predictions using the specified model file
    predictions = predict(args.model_file, df_test)

    # Save the predictions to a CSV file
    predictions.to_csv(config.TEST_PREDICTIONS_FILE, index=False)
    print(f"Predictions saved to {config.TEST_PREDICTIONS_FILE}")

    # Extract the model name from the model file path
    model_name = args.model_file.split("_")[0]

    # Visualize the model if specified in the visualize_functions dictionary
    if model_name in visualize_functions and visualize_functions[model_name] is not None:
        visualize_functions[model_name](model, df_test.drop("Segmentation", axis=1), predictions, config.VISUALS_FOLDER)
