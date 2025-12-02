# Statistical Data Analysis Tool

A comprehensive Python tool for performing statistical analysis on datasets. This tool provides functionality for calculating descriptive statistics, creating visualizations, performing correlation analysis, and conducting hypothesis tests.

## Features

### 📊 Descriptive Statistics
- **Mean**: Calculate average values
- **Median**: Find middle values
- **Mode**: Identify most frequent values
- **Standard Deviation**: Measure data spread
- **Variance**: Calculate data variability
- **Summary Statistics**: Generate comprehensive statistical summaries

### 📈 Visualizations
- **Histograms**: Visualize data distributions
- **Box Plots**: Display data quartiles and outliers
- **Correlation Heatmaps**: Show relationships between variables

### 🔗 Correlation Analysis
- Correlation matrices for all numeric variables
- Pairwise correlation calculations
- Multiple correlation methods (Pearson, Spearman, Kendall)

### 🧪 Hypothesis Testing
- **One-sample t-test**: Test if a sample mean differs from a population mean
- **Two-sample t-test**: Compare means of two independent samples
- **Chi-square test**: Test independence of categorical variables
- **ANOVA**: Compare means across multiple groups
- **Normality test**: Shapiro-Wilk test for normal distribution

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GIL794/Statistical-Data-Analysis-Tool.git
cd Statistical-Data-Analysis-Tool
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### As a Python Library

```python
from statistical_analysis import StatisticalAnalyzer
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize the analyzer
analyzer = StatisticalAnalyzer(data)

# Calculate statistics
mean_value = analyzer.mean('column_name')
median_value = analyzer.median('column_name')
std_dev = analyzer.std_deviation('column_name')

# Get summary statistics
summary = analyzer.summary_statistics()
print(summary)

# Correlation analysis
correlation = analyzer.correlation('column1', 'column2')
corr_matrix = analyzer.correlation_matrix()

# Create visualizations
analyzer.create_histogram('column_name', bins=30)
analyzer.create_box_plot(['col1', 'col2', 'col3'])
analyzer.create_correlation_heatmap()

# Hypothesis testing
t_stat, p_value = analyzer.t_test_one_sample('column_name', 100)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

### Command-Line Interface

The tool includes a CLI for quick analysis:

```bash
# Show summary statistics
python cli.py data.csv --summary

# Calculate specific statistics
python cli.py data.csv --mean column_name
python cli.py data.csv --median column_name
python cli.py data.csv --std column_name

# Show correlation matrix
python cli.py data.csv --correlation

# Create visualizations
python cli.py data.csv --histogram column_name --bins 20
python cli.py data.csv --boxplot col1 col2 col3
python cli.py data.csv --heatmap

# Hypothesis testing
python cli.py data.csv --ttest-one column_name 100
python cli.py data.csv --ttest-two column1 column2
python cli.py data.csv --normality column_name

# Save plots
python cli.py data.csv --histogram column_name --save output.png
```

### Run the Demo

A demonstration script is included to showcase all features:

```bash
python demo.py
```

This will:
- Create sample data
- Demonstrate all statistical functions
- Generate example visualizations
- Perform hypothesis tests
- Save output files for review

## Examples

### Example 1: Basic Analysis

```python
from statistical_analysis import StatisticalAnalyzer
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'scores': np.random.normal(75, 10, 100),
    'ages': np.random.randint(18, 65, 100)
})

analyzer = StatisticalAnalyzer(data)

# Get descriptive statistics
print("Mean score:", analyzer.mean('scores'))
print("Median age:", analyzer.median('ages'))
print("\nSummary:")
print(analyzer.summary_statistics())
```

### Example 2: Correlation Analysis

```python
# Calculate correlation
correlation = analyzer.correlation('scores', 'ages')
print(f"Correlation: {correlation:.3f}")

# Show correlation matrix
print(analyzer.correlation_matrix())

# Visualize correlations
analyzer.create_correlation_heatmap()
```

### Example 3: Hypothesis Testing

```python
# Test if mean score is different from 75
t_stat, p_value = analyzer.t_test_one_sample('scores', 75)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The mean is significantly different from 75")
else:
    print("No significant difference from 75")

# Test normality
stat, p_value = analyzer.normality_test('scores')
if p_value > 0.05:
    print("Data appears normally distributed")
```

## Data Format Support

The tool supports multiple data formats:
- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- **JSON** (`.json`)

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
- seaborn >= 0.11.0

## Testing

Run the test suite:

```bash
python -m unittest test_statistical_analysis.py
```

Or with verbose output:

```bash
python -m unittest test_statistical_analysis.py -v
```

## Project Structure

```
Statistical-Data-Analysis-Tool/
│
├── statistical_analysis.py    # Main analysis module
├── cli.py                      # Command-line interface
├── demo.py                     # Demonstration script
├── test_statistical_analysis.py # Unit tests
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Skills Demonstrated

This project showcases:
- **Statistics Knowledge**: Implementation of various statistical measures and tests
- **Data Manipulation**: Using pandas and numpy for data processing
- **Data Visualization**: Creating informative plots with matplotlib and seaborn
- **Software Engineering**: Clean code structure, documentation, and testing
- **Python Proficiency**: Object-oriented programming, type hints, error handling

## Author

Created as a demonstration of statistical analysis and data manipulation capabilities.
