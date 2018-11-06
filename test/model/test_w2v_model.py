from recommender.model.w2v_model import W2VModel

SENTENCES = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]


def  test_W2VModel_train():
    model = W2VModel()
    model.train_model(SENTENCES)
    vocab=model.get_model_vocabulary()
    assert len(vocab)==5

def test_split_playlist_in_history_and_groundtruth():
    model = W2VModel()
    even_list=["a","b","c","d"]
    odd_list=["e","f","g","h","i"]
    even_history, even_groundtruth = model.split_playlist_in_history_and_groundtruth(even_list)
    assert even_history ==["a","b"]
    assert even_groundtruth == ["c", "d"]
    odd_history, odd_groundtruth = model.split_playlist_in_history_and_groundtruth(odd_list)
    assert odd_history == ["e", "f", "g"]
    assert odd_groundtruth == ["h", "i"]

def test_convert_history_items_to_vectors():
    history_items = ['cat', 'dog','rabbit']
    model = W2VModel()
    model.train_model(SENTENCES)
    item_embeddings = model.convert_history_items_to_vectors(history_items)
    assert len(item_embeddings) == 2
    assert len(item_embeddings[0]) == 20
