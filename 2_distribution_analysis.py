"""
Loan Approval Dataset - Step 2: Distribution Analysis
======================================================

This script demonstrates how to:
1. Analyze the distribution of numerical features
2. Identify probability distributions (Normal, Uniform, etc.)
3. Test for normality using statistical tests
4. Visualize distributions with histograms and Q-Q plots
5. Analyze categorical variable distributions

Author: Data Analysis Lecture Series
Date: 2024
"""

# ============================================================================
# PART 1: IMPORT NECESSARY LIBRARIES
# ============================================================================

# Core data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis libraries
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest

# For creating directories and handling file paths
import os
import warnings
warnings.filterwarnings('ignore')

# Set visualization style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# PART 2: LOAD THE DATASET
# ============================================================================

def load_data(file_path):
    """
    Load the loan approval dataset from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        The path to the CSV file
        
    Returns:
    --------
    df : pandas DataFrame
        The loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print("✓ Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None


# ============================================================================
# PART 3: ANALYZE NUMERICAL DISTRIBUTIONS
# ============================================================================

def analyze_numerical_distributions(df, output_dir):
    """
    Analyze the distribution of numerical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyze
    output_dir : str
        Directory to save visualization outputs
    """
    
    print("\n" + "=" * 70)
    print("ANALYZING NUMERICAL DISTRIBUTIONS")
    print("=" * 70)
    
    # Select only numerical columns (integers and floats)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("No numerical columns found in the dataset.")
        return
    
    print(f"\nFound {len(numerical_cols)} numerical columns:")
    print(numerical_cols)
    
    # Analyze each numerical column
    for col in numerical_cols:
        print(f"\n{'-' * 70}")
        print(f"Column: {col}")
        print(f"{'-' * 70}")
        
        # Remove missing values for analysis
        data = df[col].dropna()
        
        if len(data) == 0:
            print(f"  ✗ No valid data in column '{col}'")
            continue
        
        # Calculate basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Mean:     {data.mean():.4f}")
        print(f"  Median:   {data.median():.4f}")
        print(f"  Std Dev:  {data.std():.4f}")
        print(f"  Min:      {data.min():.4f}")
        print(f"  Max:      {data.max():.4f}")
        print(f"  Skewness: {data.skew():.4f}")  # Measure of asymmetry
        print(f"  Kurtosis: {data.kurtosis():.4f}")  # Measure of tail heaviness
        
        # Perform normality tests
        print(f"\nNormality Tests:")
        
        # Shapiro-Wilk Test (works well for small to medium samples)
        # Null Hypothesis: Data is normally distributed
        # If p-value < 0.05, we reject the hypothesis (data is NOT normal)
        if len(data) <= 5000:  # Shapiro-Wilk works best with smaller samples
            stat, p_value = shapiro(data)
            print(f"  Shapiro-Wilk Test:")
            print(f"    Statistic: {stat:.4f}")
            print(f"    P-value:   {p_value:.4f}")
            if p_value > 0.05:
                print(f"    → Data appears NORMAL (p > 0.05)")
            else:
                print(f"    → Data appears NON-NORMAL (p ≤ 0.05)")
        
        # D'Agostino's K-squared Test (good for larger samples)
        if len(data) >= 8:  # Minimum sample size for this test
            stat, p_value = normaltest(data)
            print(f"  D'Agostino-Pearson Test:")
            print(f"    Statistic: {stat:.4f}")
            print(f"    P-value:   {p_value:.4f}")
            if p_value > 0.05:
                print(f"    → Data appears NORMAL (p > 0.05)")
            else:
                print(f"    → Data appears NON-NORMAL (p ≤ 0.05)")
        
        # Create visualizations for this column
        create_distribution_plots(data, col, output_dir)


# ============================================================================
# PART 4: CREATE DISTRIBUTION VISUALIZATIONS
# ============================================================================

def create_distribution_plots(data, column_name, output_dir):
    """
    Create comprehensive distribution plots for a numerical column.
    
    Parameters:
    -----------
    data : pandas Series
        The column data to visualize
    column_name : str
        Name of the column
    output_dir : str
        Directory to save the plots
    """
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Distribution Analysis: {column_name}', fontsize=16, fontweight='bold')
    
    # ========================================
    # Plot 1: Histogram with KDE (Kernel Density Estimate)
    # ========================================
    # A histogram shows the frequency of values in bins
    # KDE provides a smooth estimate of the probability density
    axes[0, 0].hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add a KDE curve
    data.plot(kind='kde', ax=axes[0, 0], color='red', linewidth=2, label='KDE')
    axes[0, 0].set_title('Histogram with Kernel Density Estimate', fontweight='bold')
    axes[0, 0].set_xlabel(column_name)
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ========================================
    # Plot 2: Box Plot (Shows quartiles and outliers)
    # ========================================
    # Box plot displays:
    # - Box: 25th to 75th percentile (IQR - Interquartile Range)
    # - Line in box: Median (50th percentile)
    # - Whiskers: Extend to 1.5 * IQR
    # - Points beyond whiskers: Potential outliers
    axes[0, 1].boxplot(data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_title('Box Plot (Outlier Detection)', fontweight='bold')
    axes[0, 1].set_ylabel(column_name)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ========================================
    # Plot 3: Q-Q Plot (Quantile-Quantile Plot)
    # ========================================
    # Q-Q plot compares data distribution to theoretical normal distribution
    # If points follow the diagonal line, data is normally distributed
    # Deviations from the line indicate non-normality
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ========================================
    # Plot 4: Violin Plot (Combination of box plot and KDE)
    # ========================================
    # Violin plot shows the distribution shape and quartiles
    # Wider sections indicate higher probability
    parts = axes[1, 1].violinplot([data], positions=[1], showmeans=True, showmedians=True)
    axes[1, 1].set_title('Violin Plot (Distribution Shape)', fontweight='bold')
    axes[1, 1].set_ylabel(column_name)
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels([column_name])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, f'distribution_{column_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {output_file}")
    
    # Close the figure to free memory
    plt.close()


# ============================================================================
# PART 5: ANALYZE CATEGORICAL DISTRIBUTIONS
# ============================================================================

def analyze_categorical_distributions(df, output_dir):
    """
    Analyze the distribution of categorical columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyze
    output_dir : str
        Directory to save visualizations
    """
    
    print("\n" + "=" * 70)
    print("ANALYZING CATEGORICAL DISTRIBUTIONS")
    print("=" * 70)
    
    # Select categorical columns (object type)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("No categorical columns found in the dataset.")
        return
    
    print(f"\nFound {len(categorical_cols)} categorical columns:")
    print(categorical_cols)
    
    # Analyze each categorical column
    for col in categorical_cols:
        print(f"\n{'-' * 70}")
        print(f"Column: {col}")
        print(f"{'-' * 70}")
        
        # Get value counts (frequency of each category)
        value_counts = df[col].value_counts()
        
        print(f"\nValue Counts:")
        print(value_counts)
        
        # Calculate proportions (percentages)
        proportions = df[col].value_counts(normalize=True) * 100
        print(f"\nProportions (%):")
        print(proportions.round(2))
        
        # Create visualization
        create_categorical_plots(df[col], col, output_dir)


# ============================================================================
# PART 6: CREATE CATEGORICAL VISUALIZATIONS
# ============================================================================

def create_categorical_plots(data, column_name, output_dir):
    """
    Create visualizations for categorical data.
    
    Parameters:
    -----------
    data : pandas Series
        The categorical column data
    column_name : str
        Name of the column
    output_dir : str
        Directory to save plots
    """
    
    # Remove missing values
    data_clean = data.dropna()
    
    if len(data_clean) == 0:
        print(f"  ✗ No valid data to plot for '{column_name}'")
        return
    
    # Get value counts
    value_counts = data_clean.value_counts()
    
    # Limit to top 10 categories if there are too many
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
        title_suffix = " (Top 10)"
    else:
        title_suffix = ""
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Categorical Distribution: {column_name}{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # ========================================
    # Plot 1: Bar Chart
    # ========================================
    value_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Bar Chart (Frequency)', fontweight='bold')
    axes[0].set_xlabel('Categories')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on top of bars
    for i, v in enumerate(value_counts.values):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # ========================================
    # Plot 2: Pie Chart
    # ========================================
    colors = plt.cm.Set3(range(len(value_counts)))
    axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Pie Chart (Proportions)', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, f'categorical_{column_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {output_file}")
    
    # Close figure
    plt.close()


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block
    """
    
    print("\n" + "=" * 70)
    print("LOAN APPROVAL DATASET - DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Define file paths
    data_file = "../data/loan_approval.csv"
    output_dir = "../outputs"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = load_data(data_file)
    
    if df is not None:
        # Analyze numerical distributions
        analyze_numerical_distributions(df, output_dir)
        
        # Analyze categorical distributions
        analyze_categorical_distributions(df, output_dir)
        
        print("\n" + "=" * 70)
        print("DISTRIBUTION ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nVisualizations saved in: {output_dir}/")
        print("\nKey Insights to Review:")
        print("1. Check Q-Q plots to identify normally distributed features")
        print("2. Look at skewness values (close to 0 = symmetric)")
        print("3. Examine box plots for outliers")
        print("4. Review categorical distributions for class imbalance")
        print("\nNext step: Run 3_data_cleaning.py to clean and process the data")
        print("=" * 70 + "\n")
    else:
        print("\n✗ Analysis could not be completed due to data loading error.\n")
