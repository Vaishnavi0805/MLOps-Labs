# Statistics Calculator - MLOps Lab

This lab demonstrates MLOps practices including version control with Git/GitHub, automated testing with pytest and unittest, and CI/CD implementation using GitHub Actions.

## Project Overview

The Statistics Calculator is a Python-based application that performs various statistical calculations on numerical datasets. Unlike basic calculators, this project focuses on statistical operations commonly used in data science and analytics.

### Features

The calculator provides the following statistical functions:

- **Mean (Average)**: Calculate the arithmetic mean of a dataset
- **Median**: Find the middle value in a sorted dataset
- **Mode**: Identify the most frequently occurring value(s)
- **Variance**: Measure the spread of data points
- **Standard Deviation**: Calculate the square root of variance
- **Comprehensive Statistics**: Get all statistics in one function call

## Project Structure
```
Github_Lab1/
├── .github/
│   └── workflows/
│       ├── pytest_action.yml      # GitHub Actions workflow for pytest
│       └── unittest_action.yml    # GitHub Actions workflow for unittest
├── data/
│   └── __init__.py                # Makes data a Python package
├── src/
│   ├── __init__.py                # Makes src a Python package
│   └── statistics_calculator.py  # Main statistics calculator module
├── test/
│   ├── __init__.py                # Makes test a Python package
│   ├── test_pytest.py             # Pytest test cases
│   └── test_unittest.py           # Unittest test cases
├── .gitignore                     # Git ignore file
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git installed on your system
- A GitHub account

### Step 1: Clone the Repository
```bash
git clone https://github.com/Vaishnavi0805/MLOps-Labs.git
cd Github_Lab1
```

### Step 2: Create and Activate Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Using the Statistics Calculator

Create a Python script or use Python interactive mode:
```python
from src import statistics_calculator

# Sample dataset
data = [10, 20, 30, 40, 50, 20, 30]

# Calculate individual statistics
mean = statistics_calculator.calculate_mean(data)
median = statistics_calculator.calculate_median(data)
mode = statistics_calculator.calculate_mode(data)
variance = statistics_calculator.calculate_variance(data)
std_dev = statistics_calculator.calculate_std_dev(data)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")

# Or get all statistics at once
all_stats = statistics_calculator.calculate_all_stats(data)
print(all_stats)
```

## Testing

This project includes comprehensive test suites using both pytest and unittest frameworks.

### Running Pytest Tests
```bash
# Run all pytest tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest test/test_pytest.py
```

### Running Unittest Tests
```bash
# Run all unittest tests
python -m unittest test.test_unittest

# Run with verbose output
python -m unittest test.test_unittest -v

# Run specific test class
python -m unittest test.test_unittest.TestStatisticsCalculator
```

### Test Coverage

The test suites cover:
- ✅ Basic statistical calculations (mean, median, mode, variance, std deviation)
- ✅ Edge cases (empty lists, single values, negative numbers)
- ✅ Error handling (non-numeric values, invalid inputs)
- ✅ Parametrized tests for multiple scenarios
- ✅ Multimodal distributions
- ✅ Comprehensive statistics function

## GitHub Actions CI/CD

This project uses GitHub Actions for continuous integration and automated testing.

### Pytest Workflow

**File**: `.github/workflows/pytest_action.yml`

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Actions:**
1. Checks out the code
2. Sets up Python 3.8
3. Installs dependencies from requirements.txt
4. Runs pytest with XML report generation
5. Uploads test results as artifacts
6. Notifies success or failure

### Unittest Workflow

**File**: `.github/workflows/unittest_action.yml`

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Actions:**
1. Checks out the code
2. Sets up Python 3.8
3. Installs dependencies
4. Runs unittest test suite
5. Notifies success or failure

### Viewing Test Results

After pushing to GitHub:
1. Go to your repository on GitHub
2. Click on the "Actions" tab
3. Select the workflow run you want to view
4. Click on the job to see detailed logs
5. Download test artifacts if needed

## Adding to Your GitHub Repository

### Step 1: Create a New Repository on GitHub

1. Go to GitHub and create a new repository
2. Name it (e.g., `mlops-statistics-calculator`)
3. Do NOT initialize with README (we already have one)
4. Click "Create repository"

### Step 2: Push Your Local Code to GitHub
```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit: Statistics Calculator with tests and CI/CD"

# Add remote repository
git remote add origin <your-repository-url>

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Verify GitHub Actions

1. After pushing, go to your repository on GitHub
2. Click on "Actions" tab
3. You should see both workflows running
4. Wait for them to complete and verify they pass ✅

## Making Changes

### Workflow for Updates

1. **Create a new branch:**
```bash
   git checkout -b feature/new-functionality
```

2. **Make your changes** to the code

3. **Run tests locally:**
```bash
   pytest
   python -m unittest test.test_unittest
```

4. **Commit your changes:**
```bash
   git add .
   git commit -m "Description of changes"
```

5. **Push to GitHub:**
```bash
   git push origin feature/new-functionality
```

6. **Create a Pull Request** on GitHub

7. **GitHub Actions will automatically run** tests on your PR

8. **Merge** once tests pass

## Key Differences from Original Lab

This implementation differs from the standard calculator lab in the following ways:

1. **Functionality**: Implements statistical operations instead of basic arithmetic
2. **Real-world Application**: Statistics are fundamental in data science and ML
3. **Complex Logic**: Mode calculation and variance require more sophisticated algorithms
4. **Enhanced Testing**: Includes parametrized tests and edge case handling
5. **Error Handling**: Comprehensive validation for empty lists and non-numeric inputs
6. **Documentation**: Detailed docstrings for all functions


## Author

**Vaishnavi Sarmalkar**
- Course: IE-7374 MLOps
- Institution: Northeastern University
