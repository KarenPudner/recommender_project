from gensim.models import Word2Vec

class W2VModel:
    model = None

    def __init__(self):
        pass

    def train_model(self,lists_of_films):
        self.model = Word2Vec(lists_of_films,size=100, window=5, min_count=1, workers=4)

    def get_model_vocabulary(self):
        return self.model.wv.vocab




