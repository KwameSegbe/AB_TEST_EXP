# ab_testing/analysis.py
import pandas as pd
import scipy.stats as stats

def run_experiment(file_path):
    """Runs an A/B test analysis."""
    df = pd.read_csv(file_path)
    
    # Ensure the dataset has 'group' (A/B) and 'metric' (conversion rate)
    group_a = df[df['group'] == 'A']['metric']
    group_b = df[df['group'] == 'B']['metric']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    result = f"T-Statistic: {t_stat:.3f}, P-Value: {p_value:.5f}"
    result += "\nStatistically significant!" if p_value < 0.05 else "\nNo significant difference."

    print(result)