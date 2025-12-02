"""
Statistical Data Analysis Tool
A comprehensive tool for performing statistical analysis on datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Union, List, Tuple, Optional


class StatisticalAnalyzer:
    """
    A class for performing statistical analysis on datasets.
    """
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray, list]):
        """
        Initialize the StatisticalAnalyzer with data.
        
        Args:
            data: Input data as DataFrame, numpy array, or list
        """
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, (np.ndarray, list)):
            self.df = pd.DataFrame(data)
        else:
            raise ValueError("Data must be a pandas DataFrame, numpy array, or list")
    
    def mean(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Calculate the mean of data.
        
        Args:
            column: Column name for DataFrame, None for all columns
            
        Returns:
            Mean value(s)
        """
        if column:
            return self.df[column].mean()
        return self.df.mean()
    
    def median(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Calculate the median of data.
        
        Args:
            column: Column name for DataFrame, None for all columns
            
        Returns:
            Median value(s)
        """
        if column:
            return self.df[column].median()
        return self.df.median()
    
    def mode(self, column: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the mode of data.
        
        Args:
            column: Column name for DataFrame, None for all columns
            
        Returns:
            Mode value(s)
        """
        if column:
            return self.df[column].mode()
        return self.df.mode()
    
    def std_deviation(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Calculate the standard deviation of data.
        
        Args:
            column: Column name for DataFrame, None for all columns
            
        Returns:
            Standard deviation value(s)
        """
        if column:
            return self.df[column].std()
        return self.df.std()
    
    def variance(self, column: Optional[str] = None) -> Union[float, pd.Series]:
        """
        Calculate the variance of data.
        
        Args:
            column: Column name for DataFrame, None for all columns
            
        Returns:
            Variance value(s)
        """
        if column:
            return self.df[column].var()
        return self.df.var()
    
    def summary_statistics(self, column: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.
        
        Args:
            column: Column name for DataFrame, None for all columns
            
        Returns:
            DataFrame with summary statistics
        """
        if column:
            return self.df[column].describe()
        return self.df.describe()
    
    def correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for all numeric columns.
        
        Returns:
            Correlation matrix as DataFrame
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return self.df[numeric_cols].corr()
    
    def correlation(self, col1: str, col2: str, method: str = 'pearson') -> float:
        """
        Calculate correlation between two columns.
        
        Args:
            col1: First column name
            col2: Second column name
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Correlation coefficient
        """
        return self.df[col1].corr(self.df[col2], method=method)
    
    def create_histogram(self, column: str, bins: int = 30, 
                        title: Optional[str] = None, 
                        save_path: Optional[str] = None) -> None:
        """
        Create a histogram for a column.
        
        Args:
            column: Column name to plot
            bins: Number of bins
            title: Plot title
            save_path: Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(title or f'Histogram of {column}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_box_plot(self, columns: Union[str, List[str]] = None, 
                       title: Optional[str] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Create a box plot for one or more columns.
        
        Args:
            columns: Column name(s) to plot, None for all numeric columns
            title: Plot title
            save_path: Path to save the figure
        """
        plt.figure(figsize=(12, 6))
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]
        
        data_to_plot = [self.df[col].dropna() for col in columns]
        plt.boxplot(data_to_plot, labels=columns)
        plt.ylabel('Values')
        plt.title(title or 'Box Plot')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of the correlation matrix.
        
        Args:
            save_path: Path to save the figure
        """
        plt.figure(figsize=(12, 10))
        corr_matrix = self.correlation_matrix()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def t_test_one_sample(self, column: str, popmean: float) -> Tuple[float, float]:
        """
        Perform one-sample t-test.
        
        Args:
            column: Column name to test
            popmean: Population mean to test against
            
        Returns:
            Tuple of (t-statistic, p-value)
        """
        data = self.df[column].dropna()
        t_stat, p_value = stats.ttest_1samp(data, popmean)
        return t_stat, p_value
    
    def t_test_two_sample(self, col1: str, col2: str, 
                         equal_var: bool = True) -> Tuple[float, float]:
        """
        Perform two-sample t-test.
        
        Args:
            col1: First column name
            col2: Second column name
            equal_var: Assume equal variances
            
        Returns:
            Tuple of (t-statistic, p-value)
        """
        data1 = self.df[col1].dropna()
        data2 = self.df[col2].dropna()
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        return t_stat, p_value
    
    def chi_square_test(self, col1: str, col2: str) -> Tuple[float, float, int, np.ndarray]:
        """
        Perform chi-square test of independence.
        
        Args:
            col1: First categorical column name
            col2: Second categorical column name
            
        Returns:
            Tuple of (chi2-statistic, p-value, degrees of freedom, expected frequencies)
        """
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2, p_value, dof, expected
    
    def normality_test(self, column: str) -> Tuple[float, float]:
        """
        Perform Shapiro-Wilk test for normality.
        
        Args:
            column: Column name to test
            
        Returns:
            Tuple of (statistic, p-value)
        """
        data = self.df[column].dropna()
        stat, p_value = stats.shapiro(data)
        return stat, p_value
    
    def anova_test(self, value_col: str, group_col: str) -> Tuple[float, float]:
        """
        Perform one-way ANOVA test.
        
        Args:
            value_col: Column with values to test
            group_col: Column with group labels
            
        Returns:
            Tuple of (F-statistic, p-value)
        """
        groups = [group[value_col].dropna() for name, group in self.df.groupby(group_col)]
        f_stat, p_value = stats.f_oneway(*groups)
        return f_stat, p_value


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        DataFrame with loaded data
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, **kwargs)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath, **kwargs)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Statistical Data Analysis Tool")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 100),
        'B': np.random.normal(80, 10, 100),
        'C': np.random.exponential(5, 100),
        'Category': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(sample_data)
    
    # Demonstrate basic statistics
    print("\n--- Basic Statistics ---")
    print(f"Mean of column A: {analyzer.mean('A'):.2f}")
    print(f"Median of column A: {analyzer.median('A'):.2f}")
    print(f"Standard Deviation of column A: {analyzer.std_deviation('A'):.2f}")
    print(f"Variance of column A: {analyzer.variance('A'):.2f}")
    
    print("\n--- Summary Statistics ---")
    print(analyzer.summary_statistics())
    
    print("\n--- Correlation Analysis ---")
    print(f"Correlation between A and B: {analyzer.correlation('A', 'B'):.3f}")
    print("\nCorrelation Matrix:")
    print(analyzer.correlation_matrix())
    
    print("\n--- Hypothesis Testing ---")
    t_stat, p_value = analyzer.t_test_one_sample('A', 100)
    print(f"One-sample t-test (A vs 100): t={t_stat:.3f}, p={p_value:.3f}")
    
    t_stat, p_value = analyzer.t_test_two_sample('A', 'B')
    print(f"Two-sample t-test (A vs B): t={t_stat:.3f}, p={p_value:.3f}")
    
    stat, p_value = analyzer.normality_test('A')
    print(f"Normality test for A: statistic={stat:.3f}, p={p_value:.3f}")
