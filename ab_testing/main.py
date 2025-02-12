# ab_testing/main.py
from ab_testing.analysis import run_experiment
from ab_testing.config import settings

if __name__ == "__main__":
    run_experiment(settings["DATA_PATH"])