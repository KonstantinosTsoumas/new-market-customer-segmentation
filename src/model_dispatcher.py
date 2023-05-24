from models import LogisticRegressionModel, DecisionTreeModel, RandomForestModel, NaiveBayesModel

models = {
    "logistic_regression": LogisticRegressionModel(),
    "decision_tree_gini": DecisionTreeModel(criterion="gini"),
    "decision_tree_entropy": DecisionTreeModel(criterion="entropy"),
    "rf": RandomForestModel(),
    "naive_bayes": NaiveBayesModel()
}

def get_model(model_name):
    """
    This function returns a model instance based on the input model name.
    Args:
        model_name: str, the name of the model to retrieve.
    Returns:
        An instance of the specified machine learning model.
    Raises:
        NotImplementedError: if the specified model is not implemented.
    """
    # check if the specified model is implemented
    if model_name not in models:
        raise NotImplementedError(f'{model_name} is not implemented. Available models are {list(models.keys())}.')

    # return an instance of the specified model
    return models[model_name]()