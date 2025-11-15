"""
Test Setup Script
=================

This script verifies that your environment is correctly set up
and all required files are in place.

Run this script before starting your lecture to ensure everything works!

Usage:
------
python TEST_SETUP.py
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - NOT FOUND")
        return False

def check_library(library_name):
    """Check if a Python library is installed."""
    try:
        __import__(library_name)
        print(f"  ✓ {library_name}")
        return True
    except ImportError:
        print(f"  ✗ {library_name} - NOT INSTALLED")
        return False

def main():
    """Run all setup checks."""
    
    print("\n" + "=" * 70)
    print("LOAN APPROVAL ANALYSIS - SETUP VERIFICATION")
    print("=" * 70)
    
    all_good = True
    
    # Check directory structure
    print("\n1. Checking Directory Structure...")
    print("-" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    directories = [
        (os.path.join(base_dir, "data"), "data/ folder"),
        (os.path.join(base_dir, "scripts"), "scripts/ folder"),
        (os.path.join(base_dir, "outputs"), "outputs/ folder"),
        (os.path.join(base_dir, "notebooks"), "notebooks/ folder"),
    ]
    
    for directory, description in directories:
        if not check_file_exists(directory, description):
            all_good = False
    
    # Check essential files
    print("\n2. Checking Essential Files...")
    print("-" * 70)
    
    essential_files = [
        (os.path.join(base_dir, "README.md"), "README.md"),
        (os.path.join(base_dir, "requirements.txt"), "requirements.txt"),
        (os.path.join(base_dir, "QUICK_START.md"), "QUICK_START.md"),
        (os.path.join(base_dir, "scripts", "1_load_and_assess.py"), "Script 1: Load and Assess"),
        (os.path.join(base_dir, "scripts", "2_distribution_analysis.py"), "Script 2: Distribution Analysis"),
        (os.path.join(base_dir, "scripts", "3_data_cleaning.py"), "Script 3: Data Cleaning"),
        (os.path.join(base_dir, "data", "sample_loan_data.csv"), "Sample dataset"),
    ]
    
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check Python libraries
    print("\n3. Checking Python Libraries...")
    print("-" * 70)
    
    required_libraries = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
    ]
    
    for library in required_libraries:
        if not check_library(library):
            all_good = False
    
    # Check for data file
    print("\n4. Checking Data Files...")
    print("-" * 70)
    
    data_dir = os.path.join(base_dir, "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(csv_files) > 0:
        print(f"  ✓ Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            print(f"    - {csv_file}")
    else:
        print("  ⚠ No CSV files found in data/ folder")
        print("    → Add your dataset or use sample_loan_data.csv")
    
    # Final summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ SETUP VERIFICATION COMPLETE - ALL CHECKS PASSED!")
        print("=" * 70)
        print("\nYou're ready to start the lecture!")
        print("\nRecommended order:")
        print("  1. Run: python scripts/1_load_and_assess.py")
        print("  2. Run: python scripts/2_distribution_analysis.py")
        print("  3. Run: python scripts/3_data_cleaning.py")
        print("\nOr use the Jupyter notebook for interactive demonstration:")
        print("  jupyter notebook notebooks/Interactive_Analysis.ipynb")
    else:
        print("✗ SETUP VERIFICATION FAILED - SOME ISSUES FOUND")
        print("=" * 70)
        print("\nPlease fix the issues above before starting.")
        print("\nTo install missing libraries, run:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
