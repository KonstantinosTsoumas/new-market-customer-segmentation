from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier

models = {
    "logistic_regression": LogisticRegression(),
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "rf": ensemble.RandomForestClassifier(),
    "naive_bayes": GaussianNB(),
    "one_vs_rest": OneVsRestClassifier(LogisticRegression())
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
