"""
Unit tests for Statistical Data Analysis Tool
"""

import unittest
import numpy as np
import pandas as pd
from statistical_analysis import StatisticalAnalyzer, load_data
import os
import tempfile


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for StatisticalAnalyzer class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            'C': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'Category': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
        })
        self.analyzer = StatisticalAnalyzer(self.test_data)
    
    def test_initialization(self):
        """Test analyzer initialization with different data types."""
        # DataFrame
        analyzer1 = StatisticalAnalyzer(self.test_data)
        self.assertIsInstance(analyzer1.df, pd.DataFrame)
        
        # NumPy array
        analyzer2 = StatisticalAnalyzer(np.array([[1, 2], [3, 4]]))
        self.assertIsInstance(analyzer2.df, pd.DataFrame)
        
        # List
        analyzer3 = StatisticalAnalyzer([[1, 2], [3, 4]])
        self.assertIsInstance(analyzer3.df, pd.DataFrame)
        
        # Invalid type should raise error
        with self.assertRaises(ValueError):
            StatisticalAnalyzer("invalid")
    
    def test_mean(self):
        """Test mean calculation."""
        mean_a = self.analyzer.mean('A')
        self.assertAlmostEqual(mean_a, 5.5, places=1)
        
        mean_b = self.analyzer.mean('B')
        self.assertAlmostEqual(mean_b, 11.0, places=1)
    
    def test_median(self):
        """Test median calculation."""
        median_a = self.analyzer.median('A')
        self.assertEqual(median_a, 5.5)
        
        median_c = self.analyzer.median('C')
        self.assertEqual(median_c, 5.5)
    
    def test_mode(self):
        """Test mode calculation."""
        mode_cat = self.analyzer.mode('Category')
        self.assertTrue('X' in mode_cat.values or 'Y' in mode_cat.values)
    
    def test_std_deviation(self):
        """Test standard deviation calculation."""
        std_a = self.analyzer.std_deviation('A')
        expected_std = np.std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ddof=1)
        self.assertAlmostEqual(std_a, expected_std, places=5)
    
    def test_variance(self):
        """Test variance calculation."""
        var_a = self.analyzer.variance('A')
        expected_var = np.var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ddof=1)
        self.assertAlmostEqual(var_a, expected_var, places=5)
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        summary = self.analyzer.summary_statistics('A')
        self.assertIn('mean', summary.index)
        self.assertIn('std', summary.index)
        self.assertIn('min', summary.index)
        self.assertIn('max', summary.index)
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        corr_matrix = self.analyzer.correlation_matrix()
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        
        # Check diagonal is 1
        self.assertAlmostEqual(corr_matrix.loc['A', 'A'], 1.0)
        
        # Check perfect positive correlation between A and B
        self.assertAlmostEqual(corr_matrix.loc['A', 'B'], 1.0, places=5)
        
        # Check perfect negative correlation between A and C
        self.assertAlmostEqual(corr_matrix.loc['A', 'C'], -1.0, places=5)
    
    def test_correlation(self):
        """Test correlation between two columns."""
        # Perfect positive correlation
        corr_ab = self.analyzer.correlation('A', 'B')
        self.assertAlmostEqual(corr_ab, 1.0, places=5)
        
        # Perfect negative correlation
        corr_ac = self.analyzer.correlation('A', 'C')
        self.assertAlmostEqual(corr_ac, -1.0, places=5)
    
    def test_t_test_one_sample(self):
        """Test one-sample t-test."""
        t_stat, p_value = self.analyzer.t_test_one_sample('A', 5.5)
        # Testing against the actual mean should give p-value close to 1
        self.assertGreater(p_value, 0.9)
        
        # Testing against a different value
        t_stat, p_value = self.analyzer.t_test_one_sample('A', 10)
        self.assertLess(p_value, 0.05)  # Should be significant
    
    def test_t_test_two_sample(self):
        """Test two-sample t-test."""
        t_stat, p_value = self.analyzer.t_test_two_sample('A', 'B')
        # A and B are perfectly correlated but have different means
        self.assertIsInstance(t_stat, float)
        self.assertIsInstance(p_value, float)
    
    def test_normality_test(self):
        """Test normality test."""
        # Create normally distributed data
        normal_data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 1000)
        })
        analyzer = StatisticalAnalyzer(normal_data)
        stat, p_value = analyzer.normality_test('normal')
        
        self.assertIsInstance(stat, float)
        self.assertIsInstance(p_value, float)
        self.assertGreater(p_value, 0.01)  # Should be normal
    
    def test_anova_test(self):
        """Test ANOVA test."""
        # Create data with groups
        data = pd.DataFrame({
            'value': [1, 2, 3, 10, 11, 12, 20, 21, 22],
            'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        })
        analyzer = StatisticalAnalyzer(data)
        f_stat, p_value = analyzer.anova_test('value', 'group')
        
        self.assertIsInstance(f_stat, float)
        self.assertIsInstance(p_value, float)
        self.assertLess(p_value, 0.05)  # Groups should be significantly different


class TestLoadData(unittest.TestCase):
    """Test cases for data loading function."""
    
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv(self):
        """Test loading CSV file."""
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.test_data.to_csv(csv_path, index=False)
        
        loaded_data = load_data(csv_path)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 3)
    
    def test_load_json(self):
        """Test loading JSON file."""
        json_path = os.path.join(self.temp_dir, 'test.json')
        self.test_data.to_json(json_path)
        
        loaded_data = load_data(json_path)
        self.assertIsInstance(loaded_data, pd.DataFrame)
    
    def test_unsupported_format(self):
        """Test loading unsupported file format."""
        with self.assertRaises(ValueError):
            load_data('test.txt')


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        analyzer = StatisticalAnalyzer(empty_df)
        self.assertIsInstance(analyzer.df, pd.DataFrame)
    
    def test_single_row(self):
        """Test with single row DataFrame."""
        single_row = pd.DataFrame({'A': [1]})
        analyzer = StatisticalAnalyzer(single_row)
        self.assertEqual(analyzer.mean('A'), 1)
    
    def test_with_nan_values(self):
        """Test handling of NaN values."""
        data_with_nan = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50]
        })
        analyzer = StatisticalAnalyzer(data_with_nan)
        
        # Mean should ignore NaN
        mean_a = analyzer.mean('A')
        self.assertAlmostEqual(mean_a, 3.0, places=1)
    
    def test_categorical_statistics(self):
        """Test statistics on categorical data."""
        cat_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A']
        })
        analyzer = StatisticalAnalyzer(cat_data)
        mode = analyzer.mode('category')
        self.assertEqual(mode.iloc[0], 'A')


if __name__ == '__main__':
    unittest.main()
