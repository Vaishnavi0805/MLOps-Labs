"""
Unittest test cases for Statistics Calculator
"""

import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import statistics_calculator


class TestStatisticsCalculator(unittest.TestCase):
    
    def test_calculate_mean(self):
        """Test mean calculation"""
        self.assertEqual(statistics_calculator.calculate_mean([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(statistics_calculator.calculate_mean([10, 20, 30]), 20.0)
        self.assertEqual(statistics_calculator.calculate_mean([5]), 5.0)
        self.assertEqual(statistics_calculator.calculate_mean([-1, 0, 1]), 0.0)
    
    def test_calculate_median(self):
        """Test median calculation"""
        self.assertEqual(statistics_calculator.calculate_median([1, 2, 3, 4, 5]), 3)
        self.assertEqual(statistics_calculator.calculate_median([1, 2, 3, 4]), 2.5)
        self.assertEqual(statistics_calculator.calculate_median([5]), 5)
        self.assertEqual(statistics_calculator.calculate_median([3, 1, 2]), 2)
    
    def test_calculate_mode(self):
        """Test mode calculation"""
        self.assertEqual(statistics_calculator.calculate_mode([1, 2, 2, 3, 4]), [2])
        self.assertEqual(statistics_calculator.calculate_mode([1, 1, 2, 2, 3]), [1, 2])
        self.assertEqual(statistics_calculator.calculate_mode([1, 2, 3, 4, 5]), [])
        self.assertEqual(statistics_calculator.calculate_mode([5, 5, 5]), [5])
    
    def test_calculate_variance(self):
        """Test variance calculation"""
        result = statistics_calculator.calculate_variance([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 2.0, places=2)
        
        result = statistics_calculator.calculate_variance([10, 10, 10])
        self.assertEqual(result, 0.0)
    
    def test_calculate_std_dev(self):
        """Test standard deviation calculation"""
        result = statistics_calculator.calculate_std_dev([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result, 1.41, places=2)
        
        result = statistics_calculator.calculate_std_dev([10, 10, 10])
        self.assertEqual(result, 0.0)
    
    def test_calculate_all_stats(self):
        """Test calculating all statistics at once"""
        numbers = [1, 2, 3, 4, 5]
        stats = statistics_calculator.calculate_all_stats(numbers)
        
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['median'], 3)
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 5)
    
    def test_empty_list_raises_error(self):
        """Test that empty lists raise ValueError"""
        with self.assertRaises(ValueError):
            statistics_calculator.calculate_mean([])
        
        with self.assertRaises(ValueError):
            statistics_calculator.calculate_median([])
        
        with self.assertRaises(ValueError):
            statistics_calculator.calculate_mode([])
    
    def test_non_numeric_raises_error(self):
        """Test that non-numeric values raise ValueError"""
        with self.assertRaises(ValueError):
            statistics_calculator.calculate_mean([1, 2, 'three'])
        
        with self.assertRaises(ValueError):
            statistics_calculator.calculate_median([1, 'two', 3])
        
        with self.assertRaises(ValueError):
            statistics_calculator.calculate_variance([1, 2, None])
    
    def test_mean_with_negative_numbers(self):
        """Test mean calculation with negative numbers"""
        self.assertEqual(statistics_calculator.calculate_mean([-5, -10, -15]), -10.0)
        self.assertEqual(statistics_calculator.calculate_mean([-1, 1]), 0.0)
    
    def test_median_with_negative_numbers(self):
        """Test median calculation with negative numbers"""
        self.assertEqual(statistics_calculator.calculate_median([-5, -3, -1]), -3)
        self.assertEqual(statistics_calculator.calculate_median([-10, -5, 0, 5]), -2.5)
    
    def test_variance_with_identical_values(self):
        """Test variance when all values are the same"""
        self.assertEqual(statistics_calculator.calculate_variance([7, 7, 7, 7]), 0.0)
    
    def test_mode_multimodal(self):
        """Test mode with multiple modes"""
        result = statistics_calculator.calculate_mode([1, 1, 2, 2, 3, 3])
        self.assertEqual(result, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()