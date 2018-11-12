from recommender.model.w2v_model import W2VModel
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

SENTENCES = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]


def  test_W2VModel_train():
    model = W2VModel()
    model.train_model(SENTENCES,2)
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

def test_convert_history_items_to_input_and_output_vectors():
    history_items = ['cat', 'dog', 'rabbit']
    model = W2VModel()
    vector_dimension=3
    model.train_model(SENTENCES, vector_dimension)
    item_embeddings = model.retrieve_input_and_output_vectors_for_item(history_items)
    assert len(item_embeddings) == 2
    assert len(item_embeddings[0]) == 3

def test_average_item_input_and_item_vectors():
    input_vector = [1,2,3,4]
    output_vector = [5,6,7,8]
    model = W2VModel()
    actual_averaged_vector = model.aggregate_item_input_and_output_vectors([input_vector,output_vector], 'average')
    expected_averaged_vector = np.array([3.0, 4.0, 5.0, 6.0])
    assert actual_averaged_vector[0] ==expected_averaged_vector[0]
    assert actual_averaged_vector[1] == expected_averaged_vector[1]
    assert actual_averaged_vector[2] == expected_averaged_vector[2]
    assert actual_averaged_vector[3] == expected_averaged_vector[3]

def test_sum_item_input_and_item_vectors():
    input_vector = [1,2,3,4]
    output_vector = [5,6,7,8]
    model = W2VModel()
    actual_averaged_vector = model.aggregate_item_input_and_output_vectors([input_vector,output_vector], 'sum')
    expected_averaged_vector = np.array([6.0, 8.0, 10.0, 12.0])
    assert actual_averaged_vector[0] ==expected_averaged_vector[0]
    assert actual_averaged_vector[1] == expected_averaged_vector[1]
    assert actual_averaged_vector[2] == expected_averaged_vector[2]
    assert actual_averaged_vector[3] == expected_averaged_vector[3]


def test_generate_user_embedding():
    history_items = ['cat', 'dog', 'rabbit']
    model = W2VModel()
    vector_dimension = 3
    model.train_model(SENTENCES, vector_dimension)
    item_embeddings_average= model.generate_user_embedding(history_items, 'average')
    assert len(item_embeddings_average) == 3


def test_predict():
    model = W2VModel()
    vector_dimension = 3
    model.train_model(SENTENCES, vector_dimension)
    cat_vector =  model.generate_user_embedding(['cat'])
    history = ['dog']
    predict = model.predict(cat_vector, history, 1)
    assert predict == ['cat']


def test_predict_matching_user_history():
    model = W2VModel()
    vector_dimension = 3
    model.train_model(SENTENCES, vector_dimension)
    cat_vector = model.generate_user_embedding(['cat'])
    history = ['cat']
    assert ['cat'] not in model.predict(cat_vector, history, 1)