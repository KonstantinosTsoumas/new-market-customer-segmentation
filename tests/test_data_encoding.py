import unittest
import pandas as pd
import os
from src.data_encoding import data_encoding

# Create the example dataset
data = {
    'col1': [1, 2, 3, 4, 5],
    'col2': ['A', 'B', 'A', 'B', 'C'],
    'col3': ['X', 'Y', 'Z', 'X', 'Y'],
    'target': [0, 1, 1, 0, 0]
}
encoded_data = None

def test_label_encoding():
    """
    Test the label encoding functionality.
    """
    encoding_strategy = "LabelEncoding"
    encoding_columns = ['col2', 'col3']
    encoded_data = data_encoding(encoding_strategy, data, encoding_columns)
    assert len(encoded_data.columns) == len(encoding_columns)

def test_one_hot_encoding():
    """
    Test the one-hot encoding functionality.
    """
    encoding_strategy = "OneHotEncoding"
    encoding_columns = ['col2', 'col3']
    encoded_data = data_encoding(encoding_strategy, data, encoding_columns)
    assert len(encoded_data.columns) == len(encoding_columns) + len(data.columns) - len(encoding_columns)

def test_target_encoding():
    """
    Test the target encoding functionality.
    """
    encoding_strategy = "TargetEncoding"
    encoding_columns = ['col2', 'col3']
    target_column = 'target'
    encoded_data = data_encoding(encoding_strategy, data, encoding_columns, target_column)
    assert len(encoded_data.columns) == len(encoding_columns)

if __name__ == '__main__':
    # Create the 'input' directory if it doesn't exist
    input_path = os.path.join(os.getcwd(), 'input')
    os.makedirs(input_path, exist_ok=True)

    # Run the tests
    test_label_encoding()
    test_one_hot_encoding()
    test_target_encoding()

    # Print a message if all tests passed
    print("All tests passed.")
