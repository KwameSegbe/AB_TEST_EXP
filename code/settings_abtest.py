# code/config.py
# PRETEST_PATH = 'code/data/pretest.csv'
# TEST_PATH = 'code/data/test.csv'

# ALPHA = 0.05
# SRM_ALPHA = 0.05
# EXPERIMENT_NAME = 'email_test'
AA_EXPERIMENT_NAME = 'AA_test'


import os

# This calculates the parent folder of 'code' → your project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
PRETEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'pretest.csv')

ALPHA = 0.05
SRM_ALPHA = 0.05
EXPERIMENT_NAME = 'email_test'

# Experiment parameters
ALPHA = 0.05  # Significance level
POWER = 0.80  # Statistical power
MDE = 0.10    # Minimum detectable effect
AB_ALPHA = 0.05

# Proportions
P1 = 0.10
P2 = P1 * (1 + P1)