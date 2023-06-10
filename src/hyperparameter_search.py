from sklearn.model_selection import RandomizedSearchCV
import config
import json

def tune_hyperparameters(cv_data, model_name, get_model_func, best_params=None):
    """
    This function performs hyperparameter tuning using Randomized Search CV.
    Args:
        cv_data: dict, the dictionary containing cross-validation data.
        model_name: str, the name of the model for hyperparameter tuning.
        get_model_func: function, the function to retrieve the model instance.
        best_params: dict, optional, pre-defined best hyperparameters.
    Returns:
        The best hyperparameters found during tuning.
    Raises:
        NotImplementedError: if the specified model is not implemented.
    """
    # Retrieve the model instance by calling the get_model_func
    model = get_model_func(model_name)

    # Define the hyperparameter grid based on the model
    param_grid = {}

    if model_name == "logistic_regression":
        param_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    elif model_name == "decision_tree_gini" or model_name == "decision_tree_entropy":
        param_grid = {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif model_name == "rf":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
    elif model_name == "naive_bayes":
        # No hyperparameters to tune for GaussianNB
        return {}
    elif model_name == "one_vs_rest":
        param_grid = {
            "estimator__C": [0.1, 1, 10],
            "estimator__penalty": ["l1", "l2"],
            "estimator__solver": ["liblinear"]
        }
    else:
        raise NotImplementedError(f'{model_name} is not implemented')

    # Extract the features (X) and target (y) from cv_data
    X = cv_data.drop("Segmentation", axis=1)
    y = cv_data["Segmentation"]

    if best_params:
        # Use the pre-defined best hyperparameters
        best_params = {k: [v] for k, v in best_params.items()}

        # Set the pre-defined best hyperparameters in the RandomizedSearchCV instance
        random_search = RandomizedSearchCV(
            model, param_distributions=best_params, cv=cv_data["kfold"].nunique(), n_iter=10, scoring="accuracy"
        )
    else:
        # Perform hyperparameter tuning using Randomized Search CV
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid, cv=cv_data["kfold"].nunique(), n_iter=10, scoring="accuracy"
        )

    random_search.fit(X, y)

    # Store the best hyperparameters found during tuning (for later use)
    best_params = random_search.best_params_

    # Save them to a JSON file
    with open(config.MODELS_FOLDER + '/best_hyperparameters.json', 'w') as file:
        json.dump(best_params, file)

    # Return the best hyperparameters
    return best_params