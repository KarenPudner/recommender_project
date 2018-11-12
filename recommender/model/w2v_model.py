from gensim.models import Word2Vec
import numpy as np

class W2VModel:
    model = None

    def __init__(self):
        pass

    def train_model(self,lists_of_films, vector_dimension):
        self.model = Word2Vec(lists_of_films, size=vector_dimension, window=5, min_count=1)

    def get_model_vocabulary(self):
        return self.model.wv.vocab


    # def evaluate(self, playlists):
    #     for playlist in playlists:
    #
    #         user_history, groundtruth = self.split_playlist_in_history_and_groundtruth(self,playlist)
    #         user_embedding = self.generate_user_embedding(user_history)
    #     #     if user_embedding is not None:
    # #         recommended_items = predict(m, user_embedding, user_history, LIST_LENGTH)
    # #         scores.update_all_metrics(groundtruth, recommended_items)
    # #
    # # return scores.get_metrics()

    def split_playlist_in_history_and_groundtruth(self,playlist):

        # If number of items odd, then give slightly more to the user history
        number_of_items_in_history = int(np.ceil(len(playlist)*0.5))
        history_items = playlist[:number_of_items_in_history]
        groundtruth_items = playlist[number_of_items_in_history:]

        return history_items, groundtruth_items

    def retrieve_input_and_output_vectors_for_item(self, history_items):
        # Also only consider items that are featured in the model vocabulary
        return [self.model.wv[v] for v in history_items if v in self.model.wv.vocab]

    def aggregate_item_input_and_output_vectors(self, item_vectors, method='average'):
        if len(item_vectors) > 0:
            if method == 'average':
                em = np.mean(item_vectors, axis=0)
                if np.isnan(em).any():
                    print("Problem")
                return em
            elif method == 'sum':
                return np.sum(item_vectors, axis=0)
            else:
                raise TypeError("Method must be specified as either 'average' or 'sum'")
        else:
            return None


    def generate_user_embedding(self, user_history, method='average'):
        item_embeddings = self.retrieve_input_and_output_vectors_for_item(user_history)
        return self.aggregate_item_input_and_output_vectors(item_embeddings, method)

    def predict(self, user_embedding, user_history, k):
        rec_list = list()
        kk = k

        # Keep generating recommendations until required number is achieved while
        # discounting all recommendations that have been seen.
        while len(rec_list) < k and kk < len(self.model.wv.vocab):
            recommended_items = self.model.wv.most_similar(positive=[user_embedding], topn=kk)
            rec_list = [i[0] for i in recommended_items if i[0] not in user_history]
            kk += 1

        return rec_list
