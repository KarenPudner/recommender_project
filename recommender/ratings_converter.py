import pandas as pd

def find_average_rating_per_customer(ratings):
    average_ratings_per_customer = ratings.groupby(['CustomerId'])['Rating'].mean()
    return average_ratings_per_customer

def convert_customer_rating_to_film_list(ratings):
    grouped_by = ratings.groupby(['CustomerId'])
    print("head")
    print(grouped_by.head())
    grouped_by=grouped_by.apply(lambda g: g[g['Rating'] >= g['Rating'].mean()])
    print("head filtered")
    print(grouped_by.head())
    return grouped_by