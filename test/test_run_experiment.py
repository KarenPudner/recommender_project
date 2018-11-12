import pandas as pd
from recommender import run_experiment

def test_run_experiment():
    list_of_films = run_experiment.run_experiment()
    assert len(list_of_films) >0
