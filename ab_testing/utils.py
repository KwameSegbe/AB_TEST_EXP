# ab_testing/utils.py
import pandas as pd

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)