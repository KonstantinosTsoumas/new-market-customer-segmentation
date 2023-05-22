import unittest
import sys
import pandas as pd
sys.path.append('../src')
from src.data_transformation import convert_to_integer, perform_binning

class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe
        self.data = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4', '5', '6']})

    def test_convert_to_integer(self):
        """
        This function tests the convert_to_integer function.
        """
        converted_data = convert_to_integer(self.data, 'A')
        self.assertEqual(converted_data['A'].dtype, int)

    def test_perform_binning(self):
        """
        This function tests the perform_binning function.
        """
        # Create a sample dataframe
        data = pd.DataFrame({'Age': [25, 35, 45, 55, 65, 75], 'Income': [50000, 60000, 70000, 80000, 90000, 100000]})

        # Define bin ranges and labels for 'Age' column
        age_bins = [20, 40, 60, 80, float('inf')]
        age_labels = ['20-39', '40-59', '60-79', '80+']

        # Perform binning on the 'Age' column
        binned_data = perform_binning(data, 'Age', age_bins, age_labels)

        # Check the 'Age' column
        expected_result = ['20-39', '20-39', '40-59', '40-59', '60-79', '60-79']
        self.assertEqual(list(binned_data['Age_Bin']), expected_result)

if __name__ == '__main__':
    unittest.main()