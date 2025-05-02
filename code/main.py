import sys
sys.path.append('./code')  # Add code folder to path if running from project root

from settings_abtest import ALPHA, SRM_ALPHA, EXPERIMENT_NAME, PRETEST_PATH, TEST_PATH
from utils import load_data, get_observed_counts,summarize_performance,plot_visits_per_day,plot_signup_rate_per_day,get_experiment_parameters
from utils import get_expected_counts, run_chi_square, run_proportions_ztest,describe_dataset,check_missing_values,calculate_sample_size_and_plot
from settings_abtest import ALPHA, POWER, MDE, P1, P2
# Load data
pretest = load_data(PRETEST_PATH)
test = load_data(TEST_PATH)

# Filter experiment
email_test = test[test['experiment'] == EXPERIMENT_NAME]

# Observed & expected counts
observed = get_observed_counts(email_test)
expected = get_expected_counts(len(email_test))


# Print the dataset stats
print("\n--- Pretest Dataset Stats ---")
describe_dataset(pretest)

print("\n--- Test Dataset Stats ---")
describe_dataset(test)

# Check for missing values
print("\n--- Missing Values ---")
check_missing_values(pretest)
check_missing_values(test)

# Check for performance of preset dataset.
print("\n--- Pretest Performance Summary ---")
summarize_performance(pretest)

# Plot_visits_per_day(pretest)
print("\n--- Pretest Visits per Day ---")
# plot_visits_per_day(pretest)

# plot_signups_per_rate_day(pretest) 
print("\n--- Pretest Signups per Day ---")
# plot_signup_rate_per_day(pretest)

# Run SRM chi-square test
chi_stats, pvalue = run_chi_square(observed, expected)

print(f'Chi-Square for SRM: {chi_stats:.3f}, P-value: {pvalue:.3f}')
if pvalue < SRM_ALPHA:
    print('Reject Ho: Sample ratio mismatch detected.')
else:
    print('Fail to reject Ho: No SRM detected.')


# Success counts for AB test
control_count = email_test[email_test['group'] == 0]['submitted'].sum()
treatment_count = email_test[email_test['group'] == 1]['submitted'].sum()

control_n = email_test[email_test['group'] == 0].shape[0]
treatment_n = email_test[email_test['group'] == 1].shape[0]


# Run proportions z-test
z_stat, pvalue_ab = run_proportions_ztest([control_count, treatment_count], [control_n, treatment_n])

print(f'Z-Stat for AB test: {z_stat:.3f}, P-value: {pvalue_ab:.3f}')
if pvalue_ab < ALPHA:
    print('Reject Ho: Statistically significant difference between control and treatment.')
else:
    print('Fail to reject Ho: No statistically significant difference between groups.')
    
#experiment parameters.
alpha, power, mde, p1, p2 = get_experiment_parameters()

# Calculate sample size and plot
print("\n--- Sample Size Calculation ---")
calculate_sample_size_and_plot(p1=P1, p2=P2, alpha=ALPHA, power=POWER)

