import pandas as pd

def find_average_rating_per_customer(ratings):
    average_ratings_per_customer = ratings.groupby(['CustomerId'])['Rating'].mean()
    return average_ratings_per_customer