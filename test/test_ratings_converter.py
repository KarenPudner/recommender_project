from recommender import ratings_converter
import pandas as pd

def test_average_rating():
    ratings = {'CustomerId': ['1', '1', '1','2','2','2'],
             'FilmId': ['4', '5', '6','20', '21','22'],
             'Rating': [4,3,2,4,5,3],
             'TimeStamp': [978302109,978302109,978302109,978302109,978302109,978302109]}

    df = pd.DataFrame.from_dict(ratings)
    average = ratings_converter.find_average_rating_per_customer(df)
    assert average[0]==3
    assert average[1] == 4


def test_convert_ratings_to_film_list():
    ratings = {'CustomerId': ['1', '1', '1','2','2','2'],
             'FilmId': ['4', '5', '6','20', '21','22'],
             'Rating': [4,3,2,4,5,3],
             'TimeStamp': [978302109,978302109,978302109,978302109,978302109,978302109]}

    df = pd.DataFrame.from_dict(ratings)

    average = ratings_converter.find_average_rating_per_customer(df)
    converted_list=ratings_converter.convert_customer_rating_to_film_list(df)
    expected = {
        '1': ['4', '5'],
        '2': ['20','21']
    }

    assert converted_list.shape==(4,4)
