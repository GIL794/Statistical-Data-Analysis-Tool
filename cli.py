"""
Command-line interface for Statistical Data Analysis Tool
"""

import argparse
import sys
from statistical_analysis import StatisticalAnalyzer, load_data


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Statistical Data Analysis Tool - Perform statistical analysis on datasets'
    )
    
    parser.add_argument('file', help='Path to data file (CSV, Excel, or JSON)')
    parser.add_argument('--summary', action='store_true', 
                       help='Show summary statistics')
    parser.add_argument('--mean', help='Calculate mean for specified column')
    parser.add_argument('--median', help='Calculate median for specified column')
    parser.add_argument('--mode', help='Calculate mode for specified column')
    parser.add_argument('--std', help='Calculate standard deviation for specified column')
    parser.add_argument('--var', help='Calculate variance for specified column')
    parser.add_argument('--correlation', action='store_true',
                       help='Show correlation matrix')
    parser.add_argument('--histogram', help='Create histogram for specified column')
    parser.add_argument('--boxplot', nargs='*',
                       help='Create box plot for specified column(s) or all numeric columns')
    parser.add_argument('--heatmap', action='store_true',
                       help='Create correlation heatmap')
    parser.add_argument('--ttest-one', nargs=2, metavar=('COLUMN', 'MEAN'),
                       help='Perform one-sample t-test')
    parser.add_argument('--ttest-two', nargs=2, metavar=('COL1', 'COL2'),
                       help='Perform two-sample t-test')
    parser.add_argument('--normality', help='Test normality for specified column')
    parser.add_argument('--bins', type=int, default=30,
                       help='Number of bins for histogram (default: 30)')
    parser.add_argument('--save', help='Path to save plots')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.file}...")
        data = load_data(args.file)
        analyzer = StatisticalAnalyzer(data)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Columns: {', '.join(data.columns)}\n")
        
        # Process commands
        if args.summary:
            print("=== Summary Statistics ===")
            print(analyzer.summary_statistics())
            print()
        
        if args.mean:
            result = analyzer.mean(args.mean)
            print(f"Mean of {args.mean}: {result:.4f}\n")
        
        if args.median:
            result = analyzer.median(args.median)
            print(f"Median of {args.median}: {result:.4f}\n")
        
        if args.mode:
            result = analyzer.mode(args.mode)
            print(f"Mode of {args.mode}:")
            print(result)
            print()
        
        if args.std:
            result = analyzer.std_deviation(args.std)
            print(f"Standard Deviation of {args.std}: {result:.4f}\n")
        
        if args.var:
            result = analyzer.variance(args.var)
            print(f"Variance of {args.var}: {result:.4f}\n")
        
        if args.correlation:
            print("=== Correlation Matrix ===")
            print(analyzer.correlation_matrix())
            print()
        
        if args.histogram:
            print(f"Creating histogram for {args.histogram}...")
            analyzer.create_histogram(args.histogram, bins=args.bins, save_path=args.save)
        
        if args.boxplot is not None:
            columns = args.boxplot if args.boxplot else None
            print(f"Creating box plot...")
            analyzer.create_box_plot(columns=columns, save_path=args.save)
        
        if args.heatmap:
            print("Creating correlation heatmap...")
            analyzer.create_correlation_heatmap(save_path=args.save)
        
        if args.ttest_one:
            column, mean = args.ttest_one
            mean = float(mean)
            t_stat, p_value = analyzer.t_test_one_sample(column, mean)
            print(f"=== One-Sample T-Test ===")
            print(f"Column: {column}, Population Mean: {mean}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} (α=0.05)\n")
        
        if args.ttest_two:
            col1, col2 = args.ttest_two
            t_stat, p_value = analyzer.t_test_two_sample(col1, col2)
            print(f"=== Two-Sample T-Test ===")
            print(f"Columns: {col1} vs {col2}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} (α=0.05)\n")
        
        if args.normality:
            stat, p_value = analyzer.normality_test(args.normality)
            print(f"=== Normality Test (Shapiro-Wilk) ===")
            print(f"Column: {args.normality}")
            print(f"Statistic: {stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Result: {'Data is NOT normally distributed' if p_value < 0.05 else 'Data appears normally distributed'} (α=0.05)\n")
        
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Column {e} not found in dataset.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
