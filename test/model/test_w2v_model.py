from recommender.model.w2v_model import W2VModel

SENTENCES = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]

def  test_W2VModel_train():
    model = W2VModel()
    model.train_model(SENTENCES)
    vocab=model.get_model_vocabulary()
    assert len(vocab)==5