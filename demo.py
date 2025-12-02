"""
Demonstration script for Statistical Data Analysis Tool
Shows various features and capabilities
"""

import numpy as np
import pandas as pd
from statistical_analysis import StatisticalAnalyzer
import matplotlib.pyplot as plt

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 200
    
    data = pd.DataFrame({
        'Height_cm': np.random.normal(170, 10, n_samples),
        'Weight_kg': np.random.normal(70, 12, n_samples),
        'Age': np.random.randint(20, 60, n_samples),
        'Income': np.random.exponential(50000, n_samples) + 30000,
        'Score': np.random.normal(75, 10, n_samples),
        'Group': np.random.choice(['A', 'B', 'C'], n_samples),
        'Passed': np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    })
    
    # Add some correlation
    data['BMI'] = data['Weight_kg'] / ((data['Height_cm'] / 100) ** 2)
    
    return data


def main():
    """Run the demonstration."""
    print("=" * 70)
    print("Statistical Data Analysis Tool - Demonstration")
    print("=" * 70)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    data = create_sample_dataset()
    print(f"   Dataset created with {len(data)} samples and {len(data.columns)} columns")
    print(f"   Columns: {', '.join(data.columns)}")
    
    # Save sample data
    data.to_csv('sample_data.csv', index=False)
    print("   Sample data saved to 'sample_data.csv'")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(data)
    
    # Basic Statistics
    print("\n2. Basic Statistics")
    print("-" * 70)
    print("\n   Summary Statistics:")
    print(analyzer.summary_statistics())
    
    print("\n   Specific Statistics for Height:")
    print(f"   - Mean: {analyzer.mean('Height_cm'):.2f} cm")
    print(f"   - Median: {analyzer.median('Height_cm'):.2f} cm")
    print(f"   - Mode: {analyzer.mode('Height_cm').iloc[0]:.2f} cm")
    print(f"   - Standard Deviation: {analyzer.std_deviation('Height_cm'):.2f} cm")
    print(f"   - Variance: {analyzer.variance('Height_cm'):.2f} cm²")
    
    # Correlation Analysis
    print("\n3. Correlation Analysis")
    print("-" * 70)
    print("\n   Correlation Matrix:")
    corr_matrix = analyzer.correlation_matrix()
    print(corr_matrix)
    
    print("\n   Strong correlations:")
    print(f"   - Height vs Weight: {analyzer.correlation('Height_cm', 'Weight_kg'):.3f}")
    print(f"   - Weight vs BMI: {analyzer.correlation('Weight_kg', 'BMI'):.3f}")
    
    # Visualizations
    print("\n4. Creating Visualizations")
    print("-" * 70)
    
    print("\n   a) Histogram of Height distribution...")
    analyzer.create_histogram('Height_cm', bins=20, 
                             title='Distribution of Height',
                             save_path='histogram_height.png')
    plt.close()
    print("      Saved as 'histogram_height.png'")
    
    print("\n   b) Box plots for numerical variables...")
    analyzer.create_box_plot(columns=['Height_cm', 'Weight_kg', 'Age', 'Score'],
                            title='Box Plots of Main Variables',
                            save_path='boxplot_comparison.png')
    plt.close()
    print("      Saved as 'boxplot_comparison.png'")
    
    print("\n   c) Correlation heatmap...")
    analyzer.create_correlation_heatmap(save_path='correlation_heatmap.png')
    plt.close()
    print("      Saved as 'correlation_heatmap.png'")
    
    # Hypothesis Testing
    print("\n5. Hypothesis Testing")
    print("-" * 70)
    
    print("\n   a) One-sample t-test (Is average height significantly different from 170cm?)")
    t_stat, p_value = analyzer.t_test_one_sample('Height_cm', 170)
    print(f"      T-statistic: {t_stat:.4f}")
    print(f"      P-value: {p_value:.4f}")
    print(f"      Conclusion: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} (α=0.05)")
    
    print("\n   b) Normality test for Height (Shapiro-Wilk test)")
    stat, p_value = analyzer.normality_test('Height_cm')
    print(f"      Statistic: {stat:.4f}")
    print(f"      P-value: {p_value:.4f}")
    print(f"      Conclusion: Data {'is NOT' if p_value < 0.05 else 'appears'} normally distributed")
    
    # Group analysis
    print("\n   c) Comparing groups (ANOVA test on Score by Group)")
    f_stat, p_value = analyzer.anova_test('Score', 'Group')
    print(f"      F-statistic: {f_stat:.4f}")
    print(f"      P-value: {p_value:.4f}")
    print(f"      Conclusion: Groups {'are' if p_value < 0.05 else 'are NOT'} significantly different (α=0.05)")
    
    # Advanced Analysis
    print("\n6. Advanced Analysis")
    print("-" * 70)
    
    print("\n   Data insights:")
    print(f"   - Total samples: {len(data)}")
    print(f"   - Pass rate: {(data['Passed'].sum() / len(data) * 100):.1f}%")
    print(f"   - Average BMI: {analyzer.mean('BMI'):.2f}")
    print(f"   - Income range: ${analyzer.df['Income'].min():.2f} - ${analyzer.df['Income'].max():.2f}")
    
    print("\n   Group statistics (Score by Group):")
    for group in ['A', 'B', 'C']:
        group_data = data[data['Group'] == group]['Score']
        print(f"   - Group {group}: Mean={group_data.mean():.2f}, Std={group_data.std():.2f}, N={len(group_data)}")
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - sample_data.csv")
    print("  - histogram_height.png")
    print("  - boxplot_comparison.png")
    print("  - correlation_heatmap.png")
    print("\nYou can analyze your own data using the CLI:")
    print("  python cli.py sample_data.csv --summary --correlation")
    print("=" * 70)


if __name__ == "__main__":
    main()
