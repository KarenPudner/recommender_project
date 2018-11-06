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


def test_filter_out_films_below_average_rating():
    ratings = {'CustomerId': ['1', '1', '1','2','2','2'],
             'FilmId': ['4', '5', '6','20', '21','22'],
             'Rating': [4,3,2,4,5,3],
             'TimeStamp': [978302109,978302109,978302109,978302109,978302109,978302109]}

    df = pd.DataFrame.from_dict(ratings)

    filtered=ratings_converter.filter_out_films_below_average_rating(df)

    assert filtered.shape==(4,4)
    assert filtered.iloc[0]['FilmId']=='4'
    assert filtered.shape == (4, 4)
    assert filtered.iloc[1]['FilmId'] == '5'
    assert filtered.shape == (4, 4)
    assert filtered.iloc[2]['FilmId'] == '20'
    assert filtered.shape == (4, 4)
    assert filtered.iloc[3]['FilmId'] == '21'


def test_convert_grouped_film_to_list_of_lists():
    filtered_ratings={'CustomerId': ['1', '1', '2','2'],
                    'FilmId': ['4', '5','20', '21'],
                    'Rating': [4,3,4,5],
                    'TimeStamp': [978302109,978302109,978302109,978302109]}
    df = pd.DataFrame.from_dict(filtered_ratings)

    film_list_of_lists=ratings_converter.convert_filtered_ratings_to_film_list_of_lists(df)
    assert film_list_of_lists[0]==['4','5']
    assert film_list_of_lists[1] == ['20', '21']