# code/utils.py

import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.proportion import proportions_ztest

def get_observed_counts(df, group_col='group'):
    return df[group_col].value_counts().sort_index().values

def get_expected_counts(total_count, num_groups=2):
    return np.array([total_count / num_groups] * num_groups)

def run_chi_square(observed, expected):
    return chisquare(f_obs=observed, f_exp=expected)

def run_proportions_ztest(success_counts, nobs):
    return proportions_ztest(success_counts, nobs)

def load_data(filepath):
    import pandas as pd
    return pd.read_csv(filepath)
