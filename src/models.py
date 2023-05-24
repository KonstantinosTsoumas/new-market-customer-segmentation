from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

class LogisticRegressionModel:
    """
    Wrapper class for LogisticRegression model from scikit-learn.
    """
    def __init__(self, C=1.0, max_iter=100, penalty='l2', solver='lbfgs'):
        """
        This function initializes the LogisticRegression model.
        Args:
            C: float, inverse of regularization strength (smaller values specify stronger regularization).
            max_iter: int, maximum number of iterations taken for the solver to converge.
            penalty: str, used to specify the norm used in the penalization. 'l1' or 'l2' is used for regularization.
            solver: str, algorithm to use in the optimization problem.
        Returns: -
        """
        self.model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)

    def fit(self, X, y):
        """
        This function fits the LogisticRegression model to the input data.

        Args:
            X: array-like of shape, the input data.
            y: array-like of shape, the target values.
        Returns: -
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        This function predicts the target values for the input data.

        Args:
            X: array-like of shape, the input data.
        Returns:
            array-like of shape, the predicted target values.
        """
        return self.model.predict(X)


class DecisionTreeModel:
    """
    Wrapper class for DecisionTreeClassifier model from scikit-learn.
    """
    def __init__(self, criterion='gini', max_depth=None, max_features='sqrt'):
        """
        This function initializes the DecisionTreeClassifier model.

        Args:
            criterion: str,the function to measure the quality of a split.
            max_depth: int or None, the maximum depth of the tree.
            max_features: int, float or {'auto', 'sqrt', 'log2'}, the number of features to consider when looking for the best split.
        Returns: -
        """

        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features)

    def fit(self, X, y):
        """
        This function fits the DecisionTreeClassifier model to the input data.

        Args:
            X: array-like of shape, the input data.
            y: array-like of shape, the target values.
        Returns: -
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        This function predicts the target values for the input data.

        Args:
            X: array-like of shape, the input data.
        Returns:
            array-like of shape, the predicted target values.
        """
        return self.model.predict(X)

class RandomForestModel:
    """
    Wrapper class for Random Forest model from scikit-learn.
    """
    def __init__(self, criterion='gini', n_estimators=100, max_depth=None, max_features='auto'):
        """
        This function initialize the Random Forest model with specified hyperparameters.
        Args:
            criterion: str, function to measure the quality of a split.
            n_estimators: int, number of trees in the forest.
            max_depth: int, maximum depth of the tree.
            max_features: int, number of features to consider when looking for the best split.
        Returns: -
        """
        self.model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)

    def fit(self, X, y):
        """
        This function fits the RandomForest Classifier model to the input data.

        Args:
            X: array-like of shape, the input data.
            y: array-like of shape, the target values.
        Returns: -
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        This function predicts the target values for the input data.

        Args:
            X: array-like of shape, the input data.
        Returns:
            array-like of shape, the predicted target values.
        """
        return self.model.predict(X)


class NaiveBayesModel:
        def __init__(self):
            self.model = GaussianNB()

        def fit(self, X, y):
            """
            This function fits the NaiveBayes Classifier model to the input data.

            Args:
                X: array-like of shape, the input data.
                y: array-like of shape, the target values.
            Returns: -
            """
            self.model.fit(X, y)

        def predict(self, X):
            """
            This function predicts the target values for the input data.

            Args:
                X: array-like of shape, the input data.
            Returns:
                array-like of shape, the predicted target values.
            """
            return self.model.predict(X)