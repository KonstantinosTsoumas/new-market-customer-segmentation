from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier

def get_logistic_regression():
    return LogisticRegression()

def get_decision_tree_gini():
    return DecisionTreeClassifier(criterion="gini")

def get_decision_tree_entropy():
    return DecisionTreeClassifier(criterion="entropy")

def get_rf():
    return RandomForestClassifier()

def get_naive_bayes():
    return GaussianNB()

def get_one_vs_rest():
    return OneVsRestClassifier(LogisticRegression())

# Modify the model_dispatcher dictionary
models = {
    "logistic_regression": get_logistic_regression,
    "decision_tree_gini": get_decision_tree_gini,
    "decision_tree_entropy": get_decision_tree_entropy,
    "rf": get_rf,
    "naive_bayes": get_naive_bayes,
    "one_vs_rest": get_one_vs_rest
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
