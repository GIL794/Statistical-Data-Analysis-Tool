"""
Streamlit Web Application for Statistical Data Analysis Tool
Interactive web interface for performing statistical analysis on datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistical_analysis import StatisticalAnalyzer
import io

# Configure page
st.set_page_config(
    page_title="Statistical Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def load_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    return pd.DataFrame({
        'Height_cm': np.random.normal(170, 10, 100),
        'Weight_kg': np.random.normal(70, 12, 100),
        'Age': np.random.randint(20, 60, 100),
        'Score': np.random.normal(75, 10, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100)
    })


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">📊 Statistical Data Analysis Tool</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for data input
    with st.sidebar:
        st.header("📁 Data Input")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Use Sample Data"]
        )
        
        df = None
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your data file",
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Supported formats: CSV, Excel, JSON"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} rows")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            df = load_sample_data()
            st.info("📊 Using sample dataset")
    
    # Main content area
    if df is not None:
        analyzer = StatisticalAnalyzer(df)
        
        # Data Overview Tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Overview", 
            "📈 Descriptive Statistics", 
            "🔗 Correlation Analysis",
            "📊 Visualizations",
            "🧪 Hypothesis Testing"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.subheader("Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        # Tab 2: Descriptive Statistics
        with tab2:
            st.subheader("Summary Statistics")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox(
                    "Select column for detailed statistics:",
                    numeric_cols
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{analyzer.mean(selected_col):.2f}")
                with col2:
                    st.metric("Median", f"{analyzer.median(selected_col):.2f}")
                with col3:
                    st.metric("Std Dev", f"{analyzer.std_deviation(selected_col):.2f}")
                with col4:
                    st.metric("Variance", f"{analyzer.variance(selected_col):.2f}")
                
                mode_val = analyzer.mode(selected_col)
                if not mode_val.empty:
                    st.info(f"📍 Mode: {mode_val.iloc[0]:.2f}")
                
                st.subheader("All Columns Summary")
                st.dataframe(analyzer.summary_statistics(), use_container_width=True)
            else:
                st.warning("No numeric columns found in the dataset.")
        
        # Tab 3: Correlation Analysis
        with tab3:
            st.subheader("Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Correlation Matrix
                st.subheader("Correlation Matrix")
                corr_matrix = analyzer.correlation_matrix()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None),
                           use_container_width=True)
                
                # Pairwise Correlation
                st.subheader("Pairwise Correlation")
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox("Select first variable:", numeric_cols, key='corr1')
                with col2:
                    var2 = st.selectbox("Select second variable:", 
                                       [col for col in numeric_cols if col != var1], 
                                       key='corr2')
                
                method = st.radio("Correlation method:", 
                                ['pearson', 'spearman', 'kendall'])
                
                if var1 and var2:
                    corr_value = analyzer.correlation(var1, var2, method=method)
                    st.metric(f"{method.capitalize()} Correlation", f"{corr_value:.4f}")
                    
                    # Interpretation
                    abs_corr = abs(corr_value)
                    if abs_corr > 0.7:
                        strength = "Strong"
                    elif abs_corr > 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"
                    
                    direction = "positive" if corr_value > 0 else "negative"
                    st.info(f"📊 {strength} {direction} correlation")
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        # Tab 4: Visualizations
        with tab4:
            st.subheader("Data Visualizations")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                viz_type = st.selectbox(
                    "Select visualization type:",
                    ["Histogram", "Box Plot", "Correlation Heatmap"]
                )
                
                if viz_type == "Histogram":
                    col = st.selectbox("Select column:", numeric_cols, key='hist_col')
                    bins = st.slider("Number of bins:", 10, 100, 30)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df[col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Histogram of {col}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                elif viz_type == "Box Plot":
                    selected_cols = st.multiselect(
                        "Select columns:",
                        numeric_cols,
                        default=numeric_cols[:min(4, len(numeric_cols))]
                    )
                    
                    if selected_cols:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        data_to_plot = [df[col].dropna() for col in selected_cols]
                        ax.boxplot(data_to_plot, labels=selected_cols)
                        ax.set_ylabel('Values')
                        ax.set_title('Box Plot Comparison')
                        plt.xticks(rotation=45, ha='right')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                
                elif viz_type == "Correlation Heatmap":
                    if len(numeric_cols) >= 2:
                        fig, ax = plt.subplots(figsize=(12, 10))
                        corr_matrix = analyzer.correlation_matrix()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                                  square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                        ax.set_title('Correlation Heatmap')
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("Need at least 2 numeric columns for correlation heatmap.")
            else:
                st.warning("No numeric columns found for visualization.")
        
        # Tab 5: Hypothesis Testing
        with tab5:
            st.subheader("Hypothesis Testing")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                test_type = st.selectbox(
                    "Select test type:",
                    ["One-Sample T-Test", "Two-Sample T-Test", "Normality Test (Shapiro-Wilk)", "ANOVA"]
                )
                
                if test_type == "One-Sample T-Test":
                    st.markdown("**Test if sample mean differs from a population mean**")
                    col = st.selectbox("Select column:", numeric_cols, key='ttest1_col')
                    pop_mean = st.number_input("Population mean:", value=0.0)
                    
                    if st.button("Run Test", key='ttest1_btn'):
                        t_stat, p_value = analyzer.t_test_one_sample(col, pop_mean)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("T-statistic", f"{t_stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        
                        alpha = 0.05
                        if p_value < alpha:
                            st.error(f"❌ Reject H₀: The sample mean is significantly different from {pop_mean} (α={alpha})")
                        else:
                            st.success(f"✅ Fail to reject H₀: The sample mean is not significantly different from {pop_mean} (α={alpha})")
                
                elif test_type == "Two-Sample T-Test":
                    st.markdown("**Test if two samples have different means**")
                    col1_name = st.selectbox("Select first column:", numeric_cols, key='ttest2_col1')
                    col2_name = st.selectbox("Select second column:", 
                                           [col for col in numeric_cols if col != col1_name],
                                           key='ttest2_col2')
                    
                    equal_var = st.checkbox("Assume equal variances", value=True)
                    
                    if st.button("Run Test", key='ttest2_btn'):
                        t_stat, p_value = analyzer.t_test_two_sample(col1_name, col2_name, equal_var)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("T-statistic", f"{t_stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        
                        alpha = 0.05
                        if p_value < alpha:
                            st.error(f"❌ Reject H₀: The means are significantly different (α={alpha})")
                        else:
                            st.success(f"✅ Fail to reject H₀: The means are not significantly different (α={alpha})")
                
                elif test_type == "Normality Test (Shapiro-Wilk)":
                    st.markdown("**Test if data follows a normal distribution**")
                    col = st.selectbox("Select column:", numeric_cols, key='norm_col')
                    
                    if st.button("Run Test", key='norm_btn'):
                        stat, p_value = analyzer.normality_test(col)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("W-statistic", f"{stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_value:.4f}")
                        
                        alpha = 0.05
                        if p_value < alpha:
                            st.error(f"❌ Reject H₀: Data is NOT normally distributed (α={alpha})")
                        else:
                            st.success(f"✅ Fail to reject H₀: Data appears normally distributed (α={alpha})")
                
                elif test_type == "ANOVA":
                    st.markdown("**Test if multiple groups have different means**")
                    
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if categorical_cols and numeric_cols:
                        value_col = st.selectbox("Select value column:", numeric_cols, key='anova_val')
                        group_col = st.selectbox("Select group column:", categorical_cols, key='anova_grp')
                        
                        if st.button("Run Test", key='anova_btn'):
                            f_stat, p_value = analyzer.anova_test(value_col, group_col)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("F-statistic", f"{f_stat:.4f}")
                            with col2:
                                st.metric("P-value", f"{p_value:.4f}")
                            
                            alpha = 0.05
                            if p_value < alpha:
                                st.error(f"❌ Reject H₀: At least one group mean is significantly different (α={alpha})")
                            else:
                                st.success(f"✅ Fail to reject H₀: All group means are not significantly different (α={alpha})")
                            
                            # Show group statistics
                            st.subheader("Group Statistics")
                            group_stats = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
                            st.dataframe(group_stats, use_container_width=True)
                    else:
                        st.warning("Need at least one categorical column and one numeric column for ANOVA.")
            else:
                st.warning("No numeric columns found for hypothesis testing.")
    
    else:
        # Landing page when no data is loaded
        st.info("👈 Please upload a data file or use sample data from the sidebar to get started.")
        
        st.markdown("""
        ### 🎯 Features
        
        This tool provides comprehensive statistical analysis capabilities:
        
        - **📋 Data Overview**: View your data structure and basic information
        - **📈 Descriptive Statistics**: Calculate mean, median, mode, standard deviation, variance
        - **🔗 Correlation Analysis**: Analyze relationships between variables
        - **📊 Visualizations**: Create histograms, box plots, and correlation heatmaps
        - **🧪 Hypothesis Testing**: Perform t-tests, ANOVA, and normality tests
        
        ### 📁 Supported File Formats
        - CSV (`.csv`)
        - Excel (`.xlsx`, `.xls`)
        - JSON (`.json`)
        
        ### 🚀 Getting Started
        1. Upload your data file using the sidebar
        2. Or use the sample data to explore features
        3. Navigate through tabs to perform different analyses
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666;">Built with Streamlit • Statistical Data Analysis Tool</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
