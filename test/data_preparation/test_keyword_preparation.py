import pandas as pd

from recommender.data_preparation import keyword_preparation


def test_filter_out_below_average_genome_scores():
    genome_scores = {'movieId': ['1', '1', '1', '2', '2', '2'],
                     'tagId': ['4', '5', '6', '20', '21', '22'],
                     'relevance': [4, 3, 2, 4, 5, 3]}

    df = pd.DataFrame.from_dict(genome_scores)
    filtered=keyword_preparation.filter_out_below_average_genome_scores(df)
    assert filtered.shape == (3, 3)
    assert filtered.iloc[0]['tagId'] == '4'
    assert filtered.iloc[1]['tagId'] == '20'
    assert filtered.iloc[2]['tagId'] == '21'


def test_convert_genome_scores_into_dictionary_by_film():
    genome_scores = {'movieId': ['1', '1', '1', '2', '2', '2'],
                     'tagId': ['4', '5', '6', '20', '21', '22'],
                     'relevance': [4, 4.5, 3.6, 4, 5, .7]}
    df = pd.DataFrame.from_dict(genome_scores)
    genome_dictionary = keyword_preparation.convert_genome_scores_into_dictionary_by_film(df)
    assert genome_dictionary['1'] == ['4','5', '6']
    assert genome_dictionary['2'] == ['20', '21', '22']