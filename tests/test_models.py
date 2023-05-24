import unittest
import sys
import numpy as np
import pandas as pd
sys.path.append('../src')
from src.models import LogisticRegressionModel, DecisionTreeModel, RandomForestModel, NaiveBayesModel

class TestModels(unittest.TestCase):
    def test_logistic_regression_model(self):
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        X_test = [[7, 8], [9, 10]]
        expected_predictions = [0, 1]

        model = LogisticRegressionModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Check if predictions are not empty
        self.assertIsNotNone(predictions)
        self.assertNotEqual(len(predictions), 0)

    def test_decision_tree_model(self):
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        X_test = [[7, 8], [9, 10]]
        expected_predictions = [0, 1]

        model = DecisionTreeModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Check if predictions are not empty
        self.assertIsNotNone(predictions)
        self.assertNotEqual(len(predictions), 0)

    def test_random_forest_model(self):
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        X_test = [[7, 8], [9, 10]]
        expected_predictions = [0, 1]

        model = RandomForestModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Check if predictions are not empty
        self.assertIsNotNone(predictions)
        self.assertNotEqual(len(predictions), 0)

    def test_naive_bayes_model(self):
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        X_test = [[7, 8], [9, 10]]
        expected_predictions = [0, 1]

        model = NaiveBayesModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Check if predictions are not empty
        self.assertIsNotNone(predictions)
        self.assertNotEqual(len(predictions), 0)

if __name__ == "__main__":
    unittest.main()
