from recommender import ratings_converter
import pandas as pd

def test_average_rating():
    ratings = {'CustomerId': ['1', '1', '1','2','2','2'],
             'FilmId': ['4', '5', '6','20', '21','22'],
             'Rating': [4,3,2,4,5,3],
             'TimeStamp': [978302109,978302109,978302109,978302109,978302109,978302109]}

    average = ratings_converter.find_average_rating_per_customer(df)
    assert average[0]==3
    assert average[1] == 4
