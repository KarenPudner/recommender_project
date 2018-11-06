from gensim.models import Word2Vec
import numpy as np

class W2VModel:
    model = None

    def __init__(self):
        pass

    def train_model(self,lists_of_films):
        self.model = Word2Vec(lists_of_films, size=20, window=5, min_count=1)

    def get_model_vocabulary(self):
        return self.model.wv.vocab


    def evaluate(self, playlists):
        for playlist in playlists:

            user_history, groundtruth = self.split_playlist_in_history_and_groundtruth(self,playlist)
            user_embedding = self.generate_user_embedding(user_history)
    #     if user_embedding is not None:
    #         recommended_items = predict(m, user_embedding, user_history, LIST_LENGTH)
    #         scores.update_all_metrics(groundtruth, recommended_items)
    #
    # return scores.get_metrics()

    def split_playlist_in_history_and_groundtruth(self,playlist):

        # If number of items odd, then give slightly more to the user history
        number_of_items_in_history = int(np.ceil(len(playlist)*0.5))
        history_items = playlist[:number_of_items_in_history]
        groundtruth_items = playlist[number_of_items_in_history:]

        return history_items, groundtruth_items

    def convert_history_items_to_vectors(self, history_items):
        # Also only consider items that are featured in the model vocabulary
        return [self.model.wv[v] for v in history_items if v in self.model.wv.vocab]


    def generate_user_embedding(self, user_history):
        item_embeddings = self.convert_history_items_to_vectors(user_history)
        if len(item_embeddings) > 0:
            em = np.mean(item_embeddings, axis=0)
            if np.isnan(em).any():
                print("Problem")
                return em

        else:
            return None
