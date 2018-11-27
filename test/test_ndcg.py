import pytest
from recommender.ndcg import dcg_at_k, ndcg_at_k


def test_dcg_at_k():
    r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    assert dcg_at_k(r, 1) == 3.0
    assert dcg_at_k(r, 1, method=1) == 3.0
    assert dcg_at_k(r, 2) == 5.0
    assert dcg_at_k(r, 2, method=1) == 4.2618595071429155
    assert dcg_at_k(r, 10) == 9.6051177391888114
    assert dcg_at_k(r, 11) == 9.6051177391888114


def test_dcg_at_k_wrong_method_parameter():
    with pytest.raises(ValueError):
        dcg_at_k([0], 1, 2)

    with pytest.raises(ValueError):
        dcg_at_k([0], 1, '1')


def test_dcg_at_k_list_null():
    assert dcg_at_k([], 1) == 0.0


def test_ndcg_at_k():
    r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    assert ndcg_at_k(r, 1) == 1.0
    assert ndcg_at_k(r, 1, method=1) == 1.0
    assert ndcg_at_k(r, 2) == 0.8333333333333334
    assert ndcg_at_k(r, 2, method=1) == 0.8710490642551529
    assert ndcg_at_k(r, 10) == 0.8824943995338175
    assert ndcg_at_k(r, 11) == 0.8824943995338175

    r = [2, 1, 2, 0]
    assert ndcg_at_k(r, 4) == 0.9203032077642922
    assert ndcg_at_k(r, 4, method=1) == 0.96519546960144276

    assert ndcg_at_k([0], 1) == 0.0
    assert ndcg_at_k([1], 2) == 1.0
