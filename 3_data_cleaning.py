"""
Loan Approval Dataset - Step 3: Data Cleaning and Processing
=============================================================

This script demonstrates how to:
1. Handle missing values (imputation strategies)
2. Remove duplicate rows
3. Detect and handle outliers
4. Encode categorical variables
5. Scale/normalize numerical features
6. Save the cleaned dataset

Author: Data Analysis Lecture Series
Date: 2024
"""

# ============================================================================
# PART 1: IMPORT NECESSARY LIBRARIES
# ============================================================================

# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical methods
from scipy import stats

# For file operations
import os
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
        Path to the CSV file
        
    Returns:
    --------
    df : pandas DataFrame
        The loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print("✓ Dataset loaded successfully!")
        print(f"  Initial shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None


# ============================================================================
# PART 3: HANDLE MISSING VALUES
# ============================================================================

def handle_missing_values(df):
    """
    Handle missing values using appropriate strategies.
    
    Strategies:
    - Numerical: Fill with median (robust to outliers)
    - Categorical: Fill with mode (most frequent value)
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with potential missing values
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with missing values handled
    """
    
    print("\n" + "=" * 70)
    print("STEP 1: HANDLING MISSING VALUES")
    print("=" * 70)
    
    # Check for missing values
    missing_before = df.isnull().sum()
    total_missing = missing_before.sum()
    
    if total_missing == 0:
        print("\n✓ No missing values found. Skipping this step.")
        return df
    
    print(f"\nTotal missing values: {total_missing}")
    print("\nMissing values by column:")
    print(missing_before[missing_before > 0])
    
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Handle numerical columns
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            # Use MEDIAN for numerical columns (robust to outliers)
            median_value = df_cleaned[col].median()
            df_cleaned[col].fillna(median_value, inplace=True)
            print(f"  ✓ Filled '{col}' with median: {median_value:.2f}")
    
    # Handle categorical columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            # Use MODE (most frequent value) for categorical columns
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode_value, inplace=True)
            print(f"  ✓ Filled '{col}' with mode: {mode_value}")
    
    # Verify all missing values are handled
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"\n✓ Missing values after cleaning: {missing_after}")
    
    return df_cleaned


# ============================================================================
# PART 4: REMOVE DUPLICATE ROWS
# ============================================================================

def remove_duplicates(df):
    """
    Identify and remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset potentially containing duplicates
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with duplicates removed
    """
    
    print("\n" + "=" * 70)
    print("STEP 2: REMOVING DUPLICATE ROWS")
    print("=" * 70)
    
    # Count duplicates before removal
    duplicates_count = df.duplicated().sum()
    
    if duplicates_count == 0:
        print("\n✓ No duplicate rows found.")
        return df
    
    print(f"\nFound {duplicates_count} duplicate rows")
    
    # Remove duplicates (keep the first occurrence)
    df_cleaned = df.drop_duplicates(keep='first')
    
    print(f"✓ Removed {duplicates_count} duplicate rows")
    print(f"  New shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    
    return df_cleaned


# ============================================================================
# PART 5: HANDLE OUTLIERS
# ============================================================================

def detect_outliers_iqr(df, column):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    IQR Method:
    - Q1 = 25th percentile
    - Q3 = 75th percentile
    - IQR = Q3 - Q1
    - Lower bound = Q1 - 1.5 * IQR
    - Upper bound = Q3 + 1.5 * IQR
    - Values outside these bounds are considered outliers
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    column : str
        Name of the column to check for outliers
        
    Returns:
    --------
    outliers : pandas Series (boolean)
        True for outlier rows, False otherwise
    """
    
    # Calculate quartiles
    Q1 = df[column].quantile(0.25)  # 25th percentile
    Q3 = df[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile Range
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers, lower_bound, upper_bound


def handle_outliers(df, method='cap'):
    """
    Handle outliers in numerical columns.
    
    Methods:
    - 'cap': Cap outliers at the boundary values (Winsorization)
    - 'remove': Remove rows containing outliers
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset
    method : str
        Method to handle outliers ('cap' or 'remove')
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with outliers handled
    """
    
    print("\n" + "=" * 70)
    print("STEP 3: HANDLING OUTLIERS")
    print("=" * 70)
    print(f"Method: {method.upper()}")
    
    df_cleaned = df.copy()
    
    # Get numerical columns
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    total_outliers = 0
    
    for col in numerical_cols:
        # Detect outliers
        outliers, lower_bound, upper_bound = detect_outliers_iqr(df_cleaned, col)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            print(f"\n'{col}':")
            print(f"  Outliers found: {outlier_count}")
            print(f"  Lower bound: {lower_bound:.2f}")
            print(f"  Upper bound: {upper_bound:.2f}")
            
            if method == 'cap':
                # Cap outliers at the boundary values
                df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                print(f"  ✓ Outliers capped to bounds")
            
            total_outliers += outlier_count
    
    if method == 'remove':
        # Remove rows with outliers in any column
        mask = pd.Series([False] * len(df_cleaned))
        for col in numerical_cols:
            outliers, _, _ = detect_outliers_iqr(df_cleaned, col)
            mask = mask | outliers
        
        rows_before = len(df_cleaned)
        df_cleaned = df_cleaned[~mask]
        rows_removed = rows_before - len(df_cleaned)
        print(f"\n✓ Removed {rows_removed} rows containing outliers")
    
    print(f"\n✓ Total outliers handled: {total_outliers}")
    
    return df_cleaned


# ============================================================================
# PART 6: ENCODE CATEGORICAL VARIABLES
# ============================================================================

def encode_categorical_variables(df):
    """
    Encode categorical variables into numerical format.
    
    Methods:
    - Label Encoding: For ordinal categories (ordered)
    - One-Hot Encoding: For nominal categories (no order)
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset with categorical variables
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with encoded categorical variables
    encoding_map : dict
        Dictionary mapping original values to encoded values
    """
    
    print("\n" + "=" * 70)
    print("STEP 4: ENCODING CATEGORICAL VARIABLES")
    print("=" * 70)
    
    df_encoded = df.copy()
    encoding_map = {}
    
    # Get categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("\n✓ No categorical columns to encode.")
        return df_encoded, encoding_map
    
    print(f"\nFound {len(categorical_cols)} categorical columns to encode:")
    
    for col in categorical_cols:
        unique_values = df_encoded[col].nunique()
        print(f"\n'{col}': {unique_values} unique values")
        
        if unique_values == 2:
            # Binary encoding for binary categories (e.g., Yes/No, Male/Female)
            print(f"  → Using BINARY encoding")
            
            # Create mapping
            unique_vals = df_encoded[col].unique()
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(mapping)
            encoding_map[col] = mapping
            
            print(f"  Mapping: {mapping}")
            
        elif unique_values <= 10:
            # Label encoding for low-cardinality categories
            print(f"  → Using LABEL encoding")
            
            # Create mapping
            unique_vals = df_encoded[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(mapping)
            encoding_map[col] = mapping
            
            print(f"  Mapping: {mapping}")
            
        else:
            # One-Hot encoding for high-cardinality categories
            print(f"  → Using ONE-HOT encoding")
            
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            
            # Add dummy columns to dataframe
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            # Remove original column
            df_encoded = df_encoded.drop(col, axis=1)
            
            print(f"  Created {len(dummies.columns)} new columns")
    
    print(f"\n✓ Categorical encoding complete")
    print(f"  New shape: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
    
    return df_encoded, encoding_map


# ============================================================================
# PART 7: SCALE NUMERICAL FEATURES
# ============================================================================

def scale_features(df, method='standardize'):
    """
    Scale numerical features to a common range.
    
    Methods:
    - 'standardize': Z-score normalization (mean=0, std=1)
      Formula: (x - mean) / std
    - 'normalize': Min-Max scaling (range 0 to 1)
      Formula: (x - min) / (max - min)
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataset with numerical features
    method : str
        Scaling method ('standardize' or 'normalize')
        
    Returns:
    --------
    df : pandas DataFrame
        Dataset with scaled features
    scaling_params : dict
        Parameters used for scaling (for future inverse transform)
    """
    
    print("\n" + "=" * 70)
    print("STEP 5: SCALING NUMERICAL FEATURES")
    print("=" * 70)
    print(f"Method: {method.upper()}")
    
    df_scaled = df.copy()
    scaling_params = {}
    
    # Get numerical columns
    numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
    
    print(f"\nScaling {len(numerical_cols)} numerical columns:")
    
    for col in numerical_cols:
        if method == 'standardize':
            # Z-score standardization
            mean = df_scaled[col].mean()
            std = df_scaled[col].std()
            
            df_scaled[col] = (df_scaled[col] - mean) / std
            
            scaling_params[col] = {'method': 'standardize', 'mean': mean, 'std': std}
            print(f"  ✓ '{col}': Standardized (mean={mean:.2f}, std={std:.2f})")
            
        elif method == 'normalize':
            # Min-Max normalization
            min_val = df_scaled[col].min()
            max_val = df_scaled[col].max()
            
            df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
            
            scaling_params[col] = {'method': 'normalize', 'min': min_val, 'max': max_val}
            print(f"  ✓ '{col}': Normalized (min={min_val:.2f}, max={max_val:.2f})")
    
    print(f"\n✓ Feature scaling complete")
    
    return df_scaled, scaling_params


# ============================================================================
# PART 8: SAVE CLEANED DATA
# ============================================================================

def save_cleaned_data(df, output_file, encoding_map, scaling_params):
    """
    Save the cleaned dataset and metadata.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The cleaned dataset
    output_file : str
        Path to save the cleaned data
    encoding_map : dict
        Categorical encoding mapping
    scaling_params : dict
        Scaling parameters
    """
    
    print("\n" + "=" * 70)
    print("STEP 6: SAVING CLEANED DATA")
    print("=" * 70)
    
    # Save the cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"\n✓ Cleaned dataset saved: {output_file}")
    print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Save encoding map
    if encoding_map:
        import json
        encoding_file = output_file.replace('.csv', '_encoding_map.json')
        with open(encoding_file, 'w') as f:
            json.dump(encoding_map, f, indent=2)
        print(f"✓ Encoding map saved: {encoding_file}")
    
    # Save scaling parameters
    if scaling_params:
        import json
        scaling_file = output_file.replace('.csv', '_scaling_params.json')
        # Convert to JSON-serializable format
        json_params = {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                          for kk, vv in v.items()} 
                      for k, v in scaling_params.items()}
        with open(scaling_file, 'w') as f:
            json.dump(json_params, f, indent=2)
        print(f"✓ Scaling parameters saved: {scaling_file}")


# ============================================================================
# PART 9: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block
    """
    
    print("\n" + "=" * 70)
    print("LOAN APPROVAL DATASET - DATA CLEANING AND PROCESSING")
    print("=" * 70)
    
    # Define file paths
    input_file = "../data/loan_approval.csv"
    output_file = "../data/loan_approval_cleaned.csv"
    
    # Load the dataset
    df = load_data(input_file)
    
    if df is not None:
        # Step 1: Handle missing values
        df = handle_missing_values(df)
        
        # Step 2: Remove duplicates
        df = remove_duplicates(df)
        
        # Step 3: Handle outliers
        # Choose method: 'cap' or 'remove'
        df = handle_outliers(df, method='cap')
        
        # Step 4: Encode categorical variables
        df, encoding_map = encode_categorical_variables(df)
        
        # Step 5: Scale numerical features
        # Choose method: 'standardize' or 'normalize'
        df, scaling_params = scale_features(df, method='standardize')
        
        # Step 6: Save cleaned data
        save_cleaned_data(df, output_file, encoding_map, scaling_params)
        
        print("\n" + "=" * 70)
        print("DATA CLEANING COMPLETE!")
        print("=" * 70)
        print("\nSummary:")
        print(f"  Original file: {input_file}")
        print(f"  Cleaned file:  {output_file}")
        print(f"  Final shape:   {df.shape[0]} rows × {df.shape[1]} columns")
        print("\nYour data is now ready for:")
        print("  ✓ Machine Learning models")
        print("  ✓ Statistical analysis")
        print("  ✓ Further exploration")
        print("=" * 70 + "\n")
    else:
        print("\n✗ Cleaning could not be completed due to data loading error.\n")
