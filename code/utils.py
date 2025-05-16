# code/utils.py

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from statsmodels.stats.proportion import proportions_ztest,proportion_effectsize,proportions_chisquare,proportions_chisquare
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.stats.power import TTestIndPower
import statsmodels.stats.api as sm
from matplotlib.ticker import MultipleLocator
import statsmodels.api as sm




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
    cohen_D = proportion_effectsize(p1, p2)

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
        
        

def get_ab_group_metrics(test_df, experiment_name='email_test'):
    """
    Filters the test dataset for a given experiment and computes control/treatment sign-up stats.

    Returns:
        - AB_test: filtered DataFrame
        - AB_control_cnt
        - AB_treatment_cnt
        - AB_control_rate
        - AB_treatment_rate
        - AB_control_size
        - AB_treatment_size
    """
    # Filter experiment
    AB_test = test_df[test_df.experiment == experiment_name]

    # Split groups
    control_signups = AB_test[AB_test.group == 0]['submitted']
    treatment_signups = AB_test[AB_test.group == 1]['submitted']

    # Compute stats
    AB_control_cnt = control_signups.sum()
    AB_treatment_cnt = treatment_signups.sum()
    AB_control_rate = control_signups.mean()
    AB_treatment_rate = treatment_signups.mean()
    AB_control_size = control_signups.count()
    AB_treatment_size = treatment_signups.count()

    # Print summary
    print(f'Control Sign-Up Rate: {AB_control_rate:.4f}')
    print(f'Treatment Sign-Up Rate: {AB_treatment_rate:.4f}')

    return (
        AB_test,
        AB_control_cnt,
        AB_treatment_cnt,
        AB_control_rate,
        AB_treatment_rate,
        AB_control_size,
        AB_treatment_size
    )
    
    

# Plotting function for daily signup rates
def plot_daily_signup_rate_by_group(AB_test, AB_control_rate, AB_treatment_rate):
    """
    Plots daily sign-up rates for control and treatment groups during the experiment.

    Parameters:
    - AB_test: DataFrame filtered for a specific experiment
    - AB_control_rate: global average sign-up rate for control group
    - AB_treatment_rate: global average sign-up rate for treatment group
    """
    # Calculate daily signup rate
    signup_per_day = AB_test.groupby(['group', 'date'])['submitted'].mean()
    ctrl_props = signup_per_day.loc[0]
    trt_props = signup_per_day.loc[1]

    # Day range for x-axis
    exp_days = range(1, AB_test['date'].nunique() + 1)

    # Plot
    f, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_days, ctrl_props, label='Control', color='b')
    ax.plot(exp_days, trt_props, label='Treatment', color='g')
    ax.axhline(AB_control_rate, label='Global Control Prop.', linestyle='--', color='b')
    ax.axhline(AB_treatment_rate, label='Global Treatment Prop.', linestyle='--', color='g')

    ax.set_xticks(exp_days)
    ax.set_title('Email Sign Up Rates Across a 14-Day Experiment')
    ax.set_ylabel('Sign-up Rate (Proportion)')
    ax.set_xlabel('Days in the Experiment')
    ax.legend()
    plt.grid(alpha=0.2)
    plt.show()
    
    
# Function to summarize AA test results 
# This function filters the pretest data for the AA test and computes control/treatment sign-up stats.
def summarize_aa_test(pretest_df, experiment_name='AA_test'):
    """
    Filters and summarizes control and treatment group metrics for an AA test.

    Parameters:
    - pretest_df: DataFrame containing pretest experiment data
    - experiment_name: string name of the AA experiment (default: 'AA_test')

    Returns:
    - AA_test: filtered dataframe
    - AA_control_cnt, AA_treatment_cnt
    - AA_control_rate, AA_treatment_rate
    - AA_control_size, AA_treatment_size
    """
    # Filter AA test
    AA_test = pretest_df[pretest_df.experiment == experiment_name]

    # Control and Treatment groups
    AA_control = AA_test[AA_test.group == 0]['submitted']
    AA_treatment = AA_test[AA_test.group == 1]['submitted']

    # Compute stats
    AA_control_cnt = AA_control.sum()
    AA_treatment_cnt = AA_treatment.sum()
    AA_control_rate = AA_control.mean()
    AA_treatment_rate = AA_treatment.mean()
    AA_control_size = AA_control.count()
    AA_treatment_size = AA_treatment.count()

    # Print output
    print('\n-------- AA Test ----------')
    print(f'Control Sign-Up Rate: {AA_control_rate:.3f}')
    print(f'Treatment Sign-Up Rate: {AA_treatment_rate:.3f}')

    return (
        AA_test,
        AA_control_cnt,
        AA_treatment_cnt,
        AA_control_rate,
        AA_treatment_rate,
        AA_control_size,
        AA_treatment_size
    )
    

import matplotlib.pyplot as plt

def plot_aa_signup_rate_by_day(AA_test, AA_control_rate, AA_treatment_rate):
    """
    Plots daily sign-up rates for control and treatment groups in an AA test.

    Parameters:
    - AA_test: DataFrame filtered to the AA experiment (must include 'group', 'date', 'submitted')
    - AA_control_rate: overall control conversion rate (for horizontal reference line)
    - AA_treatment_rate: overall treatment conversion rate
    """

    # Daily sign-up rates by group and date
    AA_signup_per_day = AA_test.groupby(['group', 'date'])['submitted'].mean()
    AA_ctrl_props = AA_signup_per_day.loc[0]
    AA_trt_props = AA_signup_per_day.loc[1]

    # Range of days
    exp_days = range(1, AA_test['date'].nunique() + 1)

    # Plot
    f, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_days, AA_ctrl_props, label='Control', color='b')
    ax.plot(exp_days, AA_trt_props, label='Treatment', color='g')
    ax.axhline(AA_control_rate, label='Global Control Prop.', linestyle='--', color='b')
    ax.axhline(AA_treatment_rate, label='Global Treatment Prop.', linestyle='--', color='g')

    ax.set_xticks(exp_days)
    ax.set_title('AA Test')
    ax.set_ylabel('Sign-up Rate (Proportion)')
    ax.set_xlabel('Days in the Experiment')
    ax.legend()
    plt.grid(alpha=0.1)
    plt.show()
    


def run_aa_chi_square_test(AA_test, control_cnt, treatment_cnt, control_size, treatment_size, alpha=0.05):
    """
    Runs a chi-square test to validate no underlying difference in AA test groups.

    Parameters:
    - AA_test: DataFrame filtered for 'AA_test'
    - control_cnt: control sign-up count
    - treatment_cnt: treatment sign-up count
    - control_size: control sample size
    - treatment_size: treatment sample size
    - alpha: significance level (default = 0.05)

    Returns:
    - chi_stat, p_value
    """

    # Ensure date column is datetime for formatting
    AA_test['date'] = pd.to_datetime(AA_test['date'], errors='coerce')
    first_date = AA_test['date'].min().strftime('%Y-%m-%d')
    last_date = AA_test['date'].max().strftime('%Y-%m-%d')

    # Run chi-square test
    chi_stat, p_value, _ = proportions_chisquare(
        [control_cnt, treatment_cnt],
        nobs=[control_size, treatment_size]
    )

    # Print results
    print(f'\n--------- AA Test ({first_date} - {last_date}) ----------\n')
    print('Ho: The sign-up rates between blue and green are the same.')
    print('Ha: The sign-up rates between blue and green are different.\n')
    print(f'Significance level: {alpha}\n')
    print(f'Chi-Square = {chi_stat:.3f} | P-value = {p_value:.3f}')

    print('\nConclusion:')
    if p_value < alpha:
        print('Reject Ho and conclude that there is statistical significance in the difference between the two groups. Check for instrumentation errors.')
    else:
        print('Fail to reject Ho. Therefore, proceed with the AB test.')

    return chi_stat, p_value


# from statsmodels.stats.proportion import proportions_chisquare
# from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
def check_sample_ratio_mismatch(test_df, experiment_name='email_test', alpha=0.05):
    """
    Performs a chi-square goodness of fit test to check for Sample Ratio Mismatch (SRM).

    Parameters:
    - test_df: full DataFrame containing experiment logs
    - experiment_name: experiment to check (default: 'email_test')
    - alpha: significance level (default = 0.05)

    Returns:
    - chi_stat, p_value
    """

    # Filter experiment
    email_test = test_df[test_df.experiment == experiment_name]

    # Ensure valid data
    email_test['group'] = pd.to_numeric(email_test['group'], errors='coerce')

    # Observed and expected sample sizes
    observed = email_test.groupby('group')['experiment'].count().values
    expected = [email_test.shape[0] * 0.5] * 2  # 50/50 split

    # Run test
    chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    # Output
    print('\n------ A Chi-Square Test for SRM ------\n')
    print('Ho: The ratio of samples is 1:1.')
    print('Ha: The ratio of samples is not 1:1.\n')
    print(f'Significance level: {alpha}\n')
    print(f'Chi-Square = {chi_stat:.3f} | P-value = {p_value:.3f}')

    print('\nConclusion:')
    if p_value < alpha:
        print('Reject Ho and conclude that there is statistical significance in the ratio of samples not being 1:1. Therefore, there is SRM.')
    else:
        print('Fail to reject Ho. Therefore, there is no SRM.')

    return chi_stat, p_value


def run_ab_chi_square_test(AB_test, control_cnt, treatment_cnt, control_size, treatment_size, alpha=0.05):
    """
    Runs a chi-square test on control vs. treatment sign-up data.
    
    Parameters:
    - AB_test: filtered dataframe for the experiment
    - control_cnt: control sign-up count
    - treatment_cnt: treatment sign-up count
    - control_size: control sample size
    - treatment_size: treatment sample size
    - alpha: significance level (default = 0.05)

    Returns:
    - (chi_stat, p_value)
    """
    # Run chi-square test
    chi_stat, p_value, _ = proportions_chisquare(
        [control_cnt, treatment_cnt],
        nobs=[control_size, treatment_size]
    )

    # Dates
    first_date = AB_test['date'].min().strftime('%Y-%m-%d')
    last_date = AB_test['date'].max().strftime('%Y-%m-%d')

    # Print results
    print(f'\n--------- AB Test Email Sign-Ups ({first_date} - {last_date}) ---------\n')
    print('Ho: The sign-up rates between blue and green are the same.')
    print('Ha: The sign-up rates between blue and green are different.\n')
    print(f'Significance level: {alpha}\n')
    print(f'Chi-Square = {chi_stat:.3f} | P-value = {p_value:.3f}')

    print('\nConclusion:')
    if p_value < alpha:
        print('Reject Ho and conclude that there is statistical significance in the difference of sign-up rates between blue and green buttons.')
    else:
        print('Fail to reject Ho. Therefore, proceed with the AB test.')

    return chi_stat, p_value



def run_ab_ttest(AB_test, alpha=0.05):
    """
    Runs a T-Test for difference in proportions between control and treatment groups.
    
    Parameters:
    - AB_test: DataFrame filtered for a specific experiment, must contain 'group', 'submitted', and 'date'
    - alpha: significance level (default = 0.05)

    Returns:
    - t_stat, p_value
    """
    # Ensure datetime formatting
    AB_test['date'] = pd.to_datetime(AB_test['date'], errors='coerce')

    # Subset signups
    AB_control_signups = AB_test[AB_test.group == 0]['submitted']
    AB_treatment_signups = AB_test[AB_test.group == 1]['submitted']

    # Run T-test
    t_stat, p_value, _ = sm.stats.ttest_ind(
        AB_treatment_signups,
        AB_control_signups,
        alternative='two-sided',
        usevar='pooled'
    )

    # Date range
    first_date = AB_test['date'].min().strftime('%Y-%m-%d')
    last_date = AB_test['date'].max().strftime('%Y-%m-%d')

    # Print summary
    print(f'\n--------- AB Test Email Sign-Ups ({first_date} - {last_date}) ---------\n')
    print('Ho: The sign-up rates between blue and green are the same.')
    print('Ha: The sign-up rates between blue and green are different.\n')
    print(f'Significance level: {alpha}\n')
    print(f'T-Statistic = {t_stat:.3f} | P-value = {p_value:.3f}')

    print('\nConclusion:')
    if p_value < alpha:
        print('Reject Ho and conclude that there is statistical significance in the difference of sign-up rates between blue and green buttons.')
    else:
        print('Fail to reject Ho.')

    return t_stat, p_value 



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

