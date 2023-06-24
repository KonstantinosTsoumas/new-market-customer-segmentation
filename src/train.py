import numpy as np
import pandas as pd
import os
import config
import model_dispatcher
import joblib
import argparse
from sklearn.metrics import classification_report
from visualize_models import visualize_functions, plot_classification_report

def run(fold, model):
    """
    This function trains a specific model for a given model (provided from command-line input) on the folded data.
    Args:
        fold: int, the fold number to train and evaluate the model for
    Returns: -
    """

    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FOLDS_FILE)

    # training data is where kfold is not equal to provided fold (plus reset index)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # drop the kfold column (not needed anymore)
    df_train = df_train.drop("kfold", axis=1)

    # validation data is where kfold is equal to provided fold (plus reset index)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop the kfold column (not needed anymore)
    df_valid = df_valid.drop("kfold", axis=1)

    # drop the 'Segmentation' column (target) from dataframe and convert it to a numpy array
    x_train = df_train.drop("Segmentation", axis=1).values
    y_train = df_train["Segmentation"].values

    # apply the same for the test set
    x_valid = df_valid.drop("Segmentation", axis=1).values
    y_valid = df_valid["Segmentation"].values

    # fetch the model from model_dispatcher
    clf = model_dispatcher.get_model(model)

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # print classification results
    print(f"\033[1m\033[4mClassification report:\033[0m\n{classification_report(y_valid, preds)}")

    # Create the plot
    fig, ax = plot_classification_report(y_valid, preds,
                                         title=f'{model}_classification_report',
                                         figsize=(8, 6), dpi=70,
                                         target_names=["A", "B", "C", "D"])

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODELS_FOLDER, f"{model}_{fold}.bin")
    )

    # visualize the model if specified in the visualize_functions dictionary
    if model in visualize_functions and visualize_functions[model] is not None:
            visualize_functions[model](clf, x_valid, y_valid)


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    # to avoid running multiple folds in the same script
    parser = argparse.ArgumentParser()

    # add fold and model
    parser.add_argument(
        "--fold",
        type=int)

    parser.add_argument(
        "--model",
        type=str)

    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)