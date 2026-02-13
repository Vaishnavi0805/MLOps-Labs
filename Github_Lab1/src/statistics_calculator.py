"""
Statistics Calculator Module
Provides functions for statistical calculations including mean, median, mode, 
standard deviation, and variance.
"""

def calculate_mean(numbers):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers (list): List of numbers (int/float)
    
    Returns:
        float: Mean of the numbers
    
    Raises:
        ValueError: If the list is empty or contains non-numeric values
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of an empty list")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements must be numbers")
    
    return sum(numbers) / len(numbers)


def calculate_median(numbers):
    """
    Calculate the median of a list of numbers.
    
    Args:
        numbers (list): List of numbers (int/float)
    
    Returns:
        float: Median of the numbers
    
    Raises:
        ValueError: If the list is empty or contains non-numeric values
    """
    if not numbers:
        raise ValueError("Cannot calculate median of an empty list")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements must be numbers")
    
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    if n % 2 == 0:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        return sorted_numbers[n//2]


def calculate_mode(numbers):
    """
    Calculate the mode(s) of a list of numbers.
    
    Args:
        numbers (list): List of numbers (int/float)
    
    Returns:
        list: List of mode value(s). Returns empty list if no mode exists.
    
    Raises:
        ValueError: If the list is empty or contains non-numeric values
    """
    if not numbers:
        raise ValueError("Cannot calculate mode of an empty list")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements must be numbers")
    
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    
    max_frequency = max(frequency.values())
    
    # If all numbers appear only once, there's no mode
    if max_frequency == 1:
        return []
    
    modes = [num for num, freq in frequency.items() if freq == max_frequency]
    return sorted(modes)


def calculate_variance(numbers):
    """
    Calculate the variance of a list of numbers (population variance).
    
    Args:
        numbers (list): List of numbers (int/float)
    
    Returns:
        float: Variance of the numbers
    
    Raises:
        ValueError: If the list is empty or contains non-numeric values
    """
    if not numbers:
        raise ValueError("Cannot calculate variance of an empty list")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements must be numbers")
    
    mean = calculate_mean(numbers)
    squared_differences = [(x - mean) ** 2 for x in numbers]
    return sum(squared_differences) / len(numbers)


def calculate_std_dev(numbers):
    """
    Calculate the standard deviation of a list of numbers (population std dev).
    
    Args:
        numbers (list): List of numbers (int/float)
    
    Returns:
        float: Standard deviation of the numbers
    
    Raises:
        ValueError: If the list is empty or contains non-numeric values
    """
    if not numbers:
        raise ValueError("Cannot calculate standard deviation of an empty list")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements must be numbers")
    
    variance = calculate_variance(numbers)
    return variance ** 0.5


def calculate_all_stats(numbers):
    """
    Calculate all statistics at once.
    
    Args:
        numbers (list): List of numbers (int/float)
    
    Returns:
        dict: Dictionary containing all calculated statistics
    
    Raises:
        ValueError: If the list is empty or contains non-numeric values
    """
    if not numbers:
        raise ValueError("Cannot calculate statistics of an empty list")
    
    if not all(isinstance(x, (int, float)) for x in numbers):
        raise ValueError("All elements must be numbers")
    
    return {
        'mean': calculate_mean(numbers),
        'median': calculate_median(numbers),
        'mode': calculate_mode(numbers),
        'variance': calculate_variance(numbers),
        'std_dev': calculate_std_dev(numbers),
        'count': len(numbers),
        'min': min(numbers),
        'max': max(numbers)
    }