import pandas as pd
import argparse
import config
from sklearn.model_selection import KFold

def create_folds(data, n_splits):
    """
    This function performs K-fold cross-validation for a given dataset.
    Args:
        data: pandas Dataframe, the dataset to be split into folds.
    Returns:
        data: pandas Dataframe, the dataset with an additional 'kfold' column indicating fold number.
    """

    # randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class
    kf = KFold(n_splits)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f

    # return dataframe with folds
    return data

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    # and take input from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', type=int)
    args = parser.parse_args()

    # read the training data from a csv file (using config)
    df_train = pd.read_csv(config.TRAINING_TRANSFORMED, header=0)

    # create folds
    df_train_folds = create_folds(df_train, args.n_splits)

    # write the training data with folds to a new csv file
    df_train_folds.to_csv(config.TRAINING_FOLDS_FILE, index=False)