import unittest
import sys
import os
import pandas as pd
sys.path.append('../src')
from src import config
from src.correlation_analysis import perform_correlation_analysis

class CorrelationAnalysisTestCase(unittest.TestCase):
    def test_valid_dataframe(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})
        perform_correlation_analysis(df)
        output_path = os.path.join(config.INPUT_FOLDER, 'corr_matrix_after_encoding.png')
        self.assertTrue(os.path.exists(output_path))

    def test_insufficient_data(self):
        df = pd.DataFrame({'A': [1], 'B': [2]})
        perform_correlation_analysis(df)
        # Add assertions here to verify the expected behavior when there is insufficient data

    def test_boolean_columns(self):
        df = pd.DataFrame({'A': [True, False, True], 'B': [True, True, False], 'C': [False, True, False]})
        perform_correlation_analysis(df)
        output_path = os.path.join(config.INPUT_FOLDER, 'corr_matrix_after_encoding.png')
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    # Set the current working directory to the directory where the script is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the tests
    unittest.main()

