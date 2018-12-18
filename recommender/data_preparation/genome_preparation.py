

# def filter_out_below_average_genome_scores(genome_df):
#     genome_df = genome_df[genome_df['relevance'] >=genome_df['relevance'].mean()]
#     return genome_df
#
# def convert_genome_scores_into_dictionary_by_film(genome_df):
#     return dict(genome_df.groupby('movieId')['tagId'].apply(list))


def create_film_feature_vector(genome_df):
    return dict(genome_df.groupby('movieId')['relevance'].apply(list))