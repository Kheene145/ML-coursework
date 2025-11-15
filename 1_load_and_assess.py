"""
Loan Approval Dataset - Step 1: Load and Assess Data
======================================================

This script demonstrates how to:
1. Load a dataset from a CSV file
2. Perform initial data assessment
3. Check data quality (missing values, duplicates)
4. Generate basic statistics

Author: Data Analysis Lecture Series
Date: 2024
"""

# ============================================================================
# PART 1: IMPORT NECESSARY LIBRARIES
# ============================================================================

# pandas: A powerful library for data manipulation and analysis
import pandas as pd

# numpy: Library for numerical computations
import numpy as np

# warnings: To suppress any warning messages for cleaner output
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 2: LOAD THE DATASET
# ============================================================================

def load_data(file_path):
    """
    Load the loan approval dataset from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing the dataset
        
    Returns:
    --------
    df : pandas DataFrame
        The loaded dataset as a DataFrame
    """
    try:
        # Read the CSV file into a pandas DataFrame
        # A DataFrame is like a table with rows and columns
        df = pd.read_csv(file_path)
        
        print("✓ Dataset loaded successfully!")
        print(f"  File: {file_path}")
        print("-" * 70)
        
        return df
        
    except FileNotFoundError:
        # This error occurs if the file doesn't exist
        print(f"✗ Error: File '{file_path}' not found!")
        print("  Please make sure the file exists in the data/ folder.")
        return None
        
    except Exception as e:
        # Catch any other unexpected errors
        print(f"✗ Error loading data: {str(e)}")
        return None


# ============================================================================
# PART 3: ASSESS THE DATASET
# ============================================================================

def assess_data(df):
    """
    Perform comprehensive initial assessment of the dataset.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to assess
    """
    
    print("\n" + "=" * 70)
    print("STEP 1: DATASET SHAPE")
    print("=" * 70)
    
    # Get the number of rows (observations) and columns (features)
    rows, cols = df.shape
    print(f"Number of rows (observations): {rows}")
    print(f"Number of columns (features): {cols}")
    
    
    print("\n" + "=" * 70)
    print("STEP 2: COLUMN NAMES AND DATA TYPES")
    print("=" * 70)
    
    # Display information about each column
    # This shows column names, non-null counts, and data types
    print("\nDataset Information:")
    print(df.info())
    
    
    print("\n" + "=" * 70)
    print("STEP 3: FIRST FEW ROWS (Preview)")
    print("=" * 70)
    
    # Display the first 5 rows of the dataset
    # This helps us understand what the data looks like
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    
    print("\n" + "=" * 70)
    print("STEP 4: MISSING VALUES ANALYSIS")
    print("=" * 70)
    
    # Count missing values in each column
    missing_values = df.isnull().sum()
    
    # Calculate the percentage of missing values
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create a DataFrame to display missing value information
    missing_info = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_percentage.values
    })
    
    # Only show columns that have missing values
    missing_info = missing_info[missing_info['Missing_Count'] > 0]
    
    if len(missing_info) > 0:
        print("\nColumns with missing values:")
        print(missing_info.to_string(index=False))
    else:
        print("\n✓ Great! No missing values found in the dataset.")
    
    
    print("\n" + "=" * 70)
    print("STEP 5: DUPLICATE ROWS CHECK")
    print("=" * 70)
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    if duplicates > 0:
        print(f"  → {duplicates} duplicate rows found (will be handled in cleaning)")
    else:
        print("  ✓ No duplicate rows found.")
    
    
    print("\n" + "=" * 70)
    print("STEP 6: BASIC STATISTICS FOR NUMERICAL COLUMNS")
    print("=" * 70)
    
    # Generate descriptive statistics for numerical columns
    # This includes count, mean, std, min, 25%, 50%, 75%, max
    print("\nNumerical columns statistics:")
    print(df.describe())
    
    
    print("\n" + "=" * 70)
    print("STEP 7: BASIC STATISTICS FOR CATEGORICAL COLUMNS")
    print("=" * 70)
    
    # Identify categorical columns (object type)
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print("\nCategorical columns summary:")
        for col in categorical_cols:
            print(f"\n{col}:")
            # Count unique values in each categorical column
            print(f"  Unique values: {df[col].nunique()}")
            # Show the frequency of top 5 values
            print("  Top 5 values:")
            print(df[col].value_counts().head().to_string())
    else:
        print("\nNo categorical columns found in the dataset.")
    
    
    print("\n" + "=" * 70)
    print("STEP 8: DATA TYPES SUMMARY")
    print("=" * 70)
    
    # Count how many columns of each data type we have
    dtype_counts = df.dtypes.value_counts()
    print("\nData types distribution:")
    print(dtype_counts)


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block - This runs when the script is executed directly
    """
    
    print("\n" + "=" * 70)
    print("LOAN APPROVAL DATASET - INITIAL ASSESSMENT")
    print("=" * 70)
    
    # Define the path to your dataset
    # IMPORTANT: Update this path if your file has a different name
    data_file = "../data/loan_approval.csv"
    
    # Load the dataset
    df = load_data(data_file)
    
    # Only proceed if the data loaded successfully
    if df is not None:
        # Perform comprehensive assessment
        assess_data(df)
        
        print("\n" + "=" * 70)
        print("ASSESSMENT COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the output above to understand your data")
        print("2. Note any missing values or data quality issues")
        print("3. Run the next script: 2_distribution_analysis.py")
        print("=" * 70 + "\n")
    else:
        print("\n✗ Assessment could not be completed due to data loading error.")
        print("  Please check the file path and try again.\n")
