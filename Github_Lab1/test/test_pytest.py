"""
Pytest test cases for Statistics Calculator
"""

import pytest
from src import statistics_calculator


def test_calculate_mean():
    """Test mean calculation"""
    assert statistics_calculator.calculate_mean([1, 2, 3, 4, 5]) == 3.0
    assert statistics_calculator.calculate_mean([10, 20, 30]) == 20.0
    assert statistics_calculator.calculate_mean([5]) == 5.0
    assert statistics_calculator.calculate_mean([-1, 0, 1]) == 0.0


def test_calculate_median():
    """Test median calculation"""
    assert statistics_calculator.calculate_median([1, 2, 3, 4, 5]) == 3
    assert statistics_calculator.calculate_median([1, 2, 3, 4]) == 2.5
    assert statistics_calculator.calculate_median([5]) == 5
    assert statistics_calculator.calculate_median([3, 1, 2]) == 2


def test_calculate_mode():
    """Test mode calculation"""
    assert statistics_calculator.calculate_mode([1, 2, 2, 3, 4]) == [2]
    assert statistics_calculator.calculate_mode([1, 1, 2, 2, 3]) == [1, 2]
    assert statistics_calculator.calculate_mode([1, 2, 3, 4, 5]) == []
    assert statistics_calculator.calculate_mode([5, 5, 5]) == [5]


def test_calculate_variance():
    """Test variance calculation"""
    result = statistics_calculator.calculate_variance([1, 2, 3, 4, 5])
    assert round(result, 2) == 2.0
    
    result = statistics_calculator.calculate_variance([10, 10, 10])
    assert result == 0.0


def test_calculate_std_dev():
    """Test standard deviation calculation"""
    result = statistics_calculator.calculate_std_dev([1, 2, 3, 4, 5])
    assert round(result, 2) == 1.41
    
    result = statistics_calculator.calculate_std_dev([10, 10, 10])
    assert result == 0.0


def test_calculate_all_stats():
    """Test calculating all statistics at once"""
    numbers = [1, 2, 3, 4, 5]
    stats = statistics_calculator.calculate_all_stats(numbers)
    
    assert stats['mean'] == 3.0
    assert stats['median'] == 3
    assert stats['count'] == 5
    assert stats['min'] == 1
    assert stats['max'] == 5


def test_empty_list_raises_error():
    """Test that empty lists raise ValueError"""
    with pytest.raises(ValueError, match="Cannot calculate mean of an empty list"):
        statistics_calculator.calculate_mean([])
    
    with pytest.raises(ValueError, match="Cannot calculate median of an empty list"):
        statistics_calculator.calculate_median([])
    
    with pytest.raises(ValueError, match="Cannot calculate mode of an empty list"):
        statistics_calculator.calculate_mode([])


def test_non_numeric_raises_error():
    """Test that non-numeric values raise ValueError"""
    with pytest.raises(ValueError, match="All elements must be numbers"):
        statistics_calculator.calculate_mean([1, 2, 'three'])
    
    with pytest.raises(ValueError, match="All elements must be numbers"):
        statistics_calculator.calculate_median([1, 'two', 3])


# Parametrized tests for mean
@pytest.mark.parametrize("numbers, expected", [
    ([1, 2, 3], 2.0),
    ([10, 20, 30, 40], 25.0),
    ([5, 5, 5, 5], 5.0),
    ([-10, 10], 0.0),
])
def test_mean_parametrized(numbers, expected):
    """Parametrized test for mean calculation"""
    assert statistics_calculator.calculate_mean(numbers) == expected


# Parametrized tests for median
@pytest.mark.parametrize("numbers, expected", [
    ([1, 2, 3], 2),
    ([1, 2, 3, 4], 2.5),
    ([5], 5),
    ([10, 20, 30], 20),
])
def test_median_parametrized(numbers, expected):
    """Parametrized test for median calculation"""
    assert statistics_calculator.calculate_median(numbers) == expected