# tests/test_analysis.py
import unittest
import pandas as pd
from ab_testing.analysis import run_experiment

class TestABTesting(unittest.TestCase):
    def test_run_experiment(self):
        data = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'metric': [0.2, 0.3, 0.25, 0.27]})
        data.to_csv("test_data.csv", index=False)
        
        run_experiment("test_data.csv")  # Should execute without error

if __name__ == "__main__":
    unittest.main()