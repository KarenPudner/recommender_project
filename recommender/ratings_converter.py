import pandas as pd

def find_average_rating_per_customer(ratings):
    average_ratings_per_customer = ratings.groupby(['CustomerId'])['Rating'].mean()
    return average_ratings_per_customer

def filter_out_films_below_average_rating(ratings):
    grouped_by = ratings.groupby(['CustomerId'])
    grouped_by=grouped_by.apply(lambda g: g[g['Rating'] >= g['Rating'].mean()])
    return grouped_by

def convert_filtered_ratings_to_film_list_of_lists(filtered_ratings):
    filtered_ratings['FilmId'] = filtered_ratings['FilmId'].apply(str)
    film_dictionary = dict(filtered_ratings.groupby('CustomerId')['FilmId'].apply(list))
    l = list(film_dictionary.values())
    return l