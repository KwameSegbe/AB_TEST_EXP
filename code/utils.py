# code/utils.py

import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.stats.power import TTestIndPower
import statsmodels.stats.api as sm
from matplotlib.ticker import MultipleLocator


def get_observed_counts(df, group_col='group'):
    return df[group_col].value_counts().sort_index().values

def get_expected_counts(total_count, num_groups=2):
    return np.array([total_count / num_groups] * num_groups)


def describe_dataset(df):
    print('# of rows:', df.shape[0])
    
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        print('Date Range:', date_min, '-', date_max)
    else:
        print('Date column not found.')


def check_missing_values(df):
    
    """ Check for missing values in the DataFrame. """
    missing_percentage = df.isnull().sum()/len(df)*100
    print(f"Missing values (%):\n{missing_percentage}")
    return missing_percentage


def summarize_performance(df):
    """
    Prints total visitors, total signups, and signup rate from dataframe.
    Assumes columns 'visitor_id' and 'submitted' exist.
    """
    total_visitors = df['visitor_id'].count()
    total_signups = df['submitted'].sum()
    signup_rate = df['submitted'].mean().round(2)
    
    print('Total visitors count:', total_visitors)
    print('Total signup count:', total_signups)
    print('Total signup rate:', signup_rate)
    
    return total_visitors, total_signups, signup_rate


# Plotting function for visits per day
def plot_visits_per_day(df):
    """
    Plots number of signups/visits per day with mean line.
    Assumes dataframe has 'date' and 'submitted' columns.
    """
    # Set the color palette for the plot
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]

    # Count sign-ups by date
    visits_per_day = df.groupby('date')['submitted'].count()
    visits_mean = visits_per_day.mean()

    # Plot data
    plt.figure(figsize=(12, 5))
    plt.plot(visits_per_day.index, visits_per_day, color=c1, linewidth=1, label='Visits')
    plt.axhline(visits_mean, color=c2, linestyle='--', linewidth=1, alpha=0.3, label='Visits (Mean)')

    # Format plot
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Adjust interval as needed
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    plt.title('Urban Wear Visitor Count', fontsize=10, weight='bold')
    plt.ylabel('Visitors', fontsize=10)
    plt.xlabel('Date', fontsize=10)
    plt.legend()
    plt.show()

# Plotting function for signup rate per day
def plot_signup_rate_per_day(df):
    """
    Plots daily sign-up rate with mean line.
    Assumes dataframe has 'date' and 'submitted' columns.
    """
    # Set the color palette for the plot
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]

    # Count sign-ups by date
    signup_rate_per_day = df.groupby('date')['submitted'].mean()
    signup_rate_mean = signup_rate_per_day.mean()

    # Plot data
    plt.figure(figsize=(12, 5))
    plt.plot(signup_rate_per_day.index, signup_rate_per_day, color=c1, linewidth=1, label='Sign_Up_Rate')

    # Fix: Use correct mean variable
    plt.axhline(signup_rate_mean, color=c2, linestyle='--', linewidth=1, alpha=0.3, label='Sign_Up_Rate (Mean)')

    # Format plot
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Adjust interval as needed
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    plt.title('Urban Wear Sign-up Rate', fontsize=10, weight='bold')
    plt.ylabel('Sign Rate', fontsize=10)
    plt.xlabel('Date', fontsize=10)
    plt.legend()
    plt.show()


def get_experiment_parameters():
    # # Experiment parameters
    """ This function returns the parameters for the experiment.
    It includes the significance level (alpha), power, minimum detectable effect (mde),
    and the proportions for the two groups (p1 and p2).
    """
    alpha = 0.05
    power = 0.80
    mde = 0.10
    p1 = 0.10
    p2 = p1 * (1 + p1)
    
    return alpha, power, mde, p1, p2


def plot_experiment_duration_by_traffic(n, visits_mean):
    """
    Plots experiment duration vs. traffic allocation using a precomputed sample size `n`.
    Does not trigger power analysis.
    """
    alloc = np.arange(0.10, 1.1, 0.10)  # Allocation percentages from 10% to 100%
    size = round(visits_mean, -3) * alloc  # Daily sample size based on traffic
    days = np.ceil(2 * n / size)  # Duration to hit total sample size

    # Generate plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alloc, days, 'o-')

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_title('Days Required Given Traffic Allocation per Day')
    ax.set_ylabel('Experiment Duration in Days')
    ax.set_xlabel('Traffic Allocated to the Experiment per Day')

    plt.grid(alpha=0.2)
    plt.show()

    # DURATION PLOT
    alloc = np.arange(0.10, 1.1, 0.10)
    size = round(visits_mean, -3) * alloc
    days = np.ceil(2 * n / size)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alloc, days, 'o-')
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_title('Days Required Given Traffic Allocation per Day')
    ax.set_ylabel('Experiment Duration in Days')
    ax.set_xlabel('Traffic Allocated to the Experiment per Day')
    plt.grid(alpha=0.2)
    plt.show()

# # Sample size calculation and power analysis.
def calculate_sample_size_and_plot(p1, p2, alpha=0.05, power=0.80):
    """
    Calculates required sample size per group using Cohen's D and plots power analysis curve.
    """
    # Calculate effect size
    cohen_D = sm.proportion_effectsize(p1, p2)

    # Estimate sample size
    ttest_power = TTestIndPower()
    n = ttest_power.solve_power(effect_size=cohen_D, power=power, alpha=alpha)
    n = int(round(n, -3))  # Round to nearest thousand

    print(f"To detect an effect of {100 * (p2/p1 - 1):.1f}% lift from the pretest sign-up at {100 * p1:.0f}%, "
          f"the sample size per group required is {n}."
          f"\nThe total sample required in the experiment is {2 * n}.")

    # Plot power analysis
    ttest_power.plot_power(dep_var='nobs', nobs=np.arange(1000, 30000, 1000),
                           effect_size=[cohen_D], title='Power Analysis')

    plt.axhline(0.8, linestyle='--', label='Desired Power', alpha=0.5)
    plt.axvline(n, linestyle='--', color='orange', label='Sample Size', alpha=0.5)
    plt.ylabel('Statistical Power')
    plt.grid(alpha=0.08)
    plt.legend()
    plt.show()

    return n, cohen_D

def plot_experiment_duration_by_traffic_absolute(size, days):
    """
    Plots experiment duration (in days) given absolute traffic volume allocated per day.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(size, days, 'o-')

    ax.xaxis.set_major_locator(MultipleLocator(1000))  # X-axis tick every 1000

    ax.set_title('Days Required Given Traffic Allocation per Day')
    ax.set_ylabel('Experiment Duration in Days')
    ax.set_xlabel('Traffic Allocated to the Experiment per Day')

    plt.grid(alpha=0.2)
    plt.show()
    
    
def print_required_users_per_day(n, durations=[21, 14, 7]):
    """
    Prints the number of users required per day for a given list of experiment durations.
    
    Parameters:
    - n: required sample size per group
    - durations: list of experiment durations in days (default: [21, 14, 7])
    """
    total_required = n * 2  # Total users needed for both groups
    print(f"\nTotal sample size required: {total_required}")
    
    for d in durations:
        per_day = int(np.ceil(total_required / d))
        print(f'For a {d}-day experiment, {per_day} users are required per day.')


def run_chi_square(observed, expected):
    return chisquare(f_obs=observed, f_exp=expected)

def run_proportions_ztest(success_counts, nobs):
    return proportions_ztest(success_counts, nobs)

def run_chi_square(observed, expected):
    expected = expected * (observed.sum() / expected.sum())
    return chisquare(f_obs=observed, f_exp=expected)

def load_data(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

