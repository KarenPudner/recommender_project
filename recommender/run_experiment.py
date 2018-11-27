from recommender import ratings_converter
from recommender.model.w2v_model import W2VModel
from recommender.metrics import Metrics
import pandas as pd
import json

def run_experiment():
    # ratings_df = pd.read_csv('/Users/pudnek01/Documents/workspace/recommender_project/ratings.csv')
    # filtered_ratings_df = ratings_converter.filter_out_films_below_average_rating(ratings_df)
    # filtered_ratings_df.to_csv('filtered_ratings.csv')
    filtered_ratings_df = pd.read_csv('filtered_ratings.csv')
    film_list_of_lists = ratings_converter.convert_filtered_ratings_to_film_list_of_lists(filtered_ratings_df)
    short_list = film_list_of_lists[1:100]
    model = W2VModel()
    table = [['Seed Films', 'Recommended Films']]
    scores = Metrics()
    for playlist in film_list_of_lists:

        user_history, groundtruth = model.split_playlist_in_history_and_groundtruth(playlist)
        user_embedding = model.generate_user_embedding(user_history)
        if user_embedding is not None:
            recommended_items = model.predict(user_embedding, user_history, 20)
            table.append([user_history, recommended_items])
            scores.update_all_metrics(groundtruth, recommended_items)
    # headers = table.pop(0)
    # df = pd.DataFrame(table, columns=headers)
    # df.to_csv('recommendations.csv')
    recommendation_metrics = scores.get_metrics()
    with open('metrics.json', 'w') as fout:
        json.dump(recommendation_metrics, fout)
    return recommendation_metrics
