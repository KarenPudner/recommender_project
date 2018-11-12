from recommender import ratings_converter
from recommender.model.w2v_model import W2VModel
import pandas as pd

def run_experiment():
    # ratings_df = pd.read_csv('/Users/pudnek01/Documents/workspace/recommender_project/ratings.csv')
    # filtered_ratings_df = ratings_converter.filter_out_films_below_average_rating(ratings_df)
    # filtered_ratings_df.to_csv('filtered_ratings.csv')
    filtered_ratings_df = pd.read_csv('filtered_ratings.csv')
    film_list_of_lists = ratings_converter.convert_filtered_ratings_to_film_list_of_lists(filtered_ratings_df)
    model = W2VModel()
    # vector_dimension = 3
    # model.train_model(film_list_of_lists, vector_dimension)
    user_embedding = model.generate_user_embedding(["2560"])
    films = model.predict(user_embedding,[],4)
    return films