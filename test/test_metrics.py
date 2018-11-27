from recommender.metrics import Metrics, get_best_params

groundtruth = ['a', 'b']
good_rec_list = ['a', 'b']
bad_rec_list = ['c', 'd']


def test_update_all_metrics():
    metrics = Metrics()
    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=[])
    assert metrics.num_users == 0
    assert metrics.sum_user_ndcg == 0.0

    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=good_rec_list)
    assert metrics.num_users == 1
    assert metrics.sum_user_ndcg == 1.0

    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=bad_rec_list)
    assert metrics.num_users == 2
    assert metrics.sum_user_ndcg == 1.0


def test_get_current_ndcg():
    metrics = Metrics()
    assert metrics.get_current_ndcg() == 0
    assert metrics.num_users == 0
    assert metrics.sum_user_ndcg == 0

    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=good_rec_list)
    assert metrics.get_current_ndcg() == 1.0

    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=bad_rec_list)
    assert metrics.get_current_ndcg() == 0.5


def test_calculate_ndcg():
    metrics = Metrics()
    assert metrics.calculate_ndcg(groundtruth=groundtruth, rec_list=good_rec_list) == 1.0
    assert metrics.calculate_ndcg(groundtruth=groundtruth, rec_list=bad_rec_list) == 0.0


def test_update_ncgd():
    metrics = Metrics()
    metrics.update_ndcg(groundtruth=groundtruth, rec_list=good_rec_list)
    assert metrics.num_users == 1
    assert metrics.sum_user_ndcg == 1.0
    assert metrics.get_current_ndcg() == 1.0

    metrics.update_ndcg(groundtruth=groundtruth, rec_list=bad_rec_list)
    assert metrics.num_users == 2
    assert metrics.sum_user_ndcg == 1.0
    assert metrics.get_current_ndcg() == 0.5


def test_get_metrics():
    metrics = Metrics()
    expected = {
        'metrics_results': [{'name': 'nDCG',
                             'values': {'mean': 0},
                             'version': '1.0'}],
        'number_of_users_analysed': 0
    }
    response = metrics.get_metrics()
    response.pop('analysis_completion_date')
    assert response == expected

    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=good_rec_list)
    expected = {
        'metrics_results': [{'name': 'nDCG',
                             'values': {'mean': 1.0},
                             'version': '1.0'}],
        'number_of_users_analysed': 1
    }
    response = metrics.get_metrics()
    response.pop('analysis_completion_date')
    assert response == expected

    metrics.update_all_metrics(groundtruth=groundtruth, rec_list=bad_rec_list)
    expected = {
        'metrics_results': [{'name': 'nDCG',
                             'values': {'mean': 0.5},
                             'version': '1.0'}],
        'number_of_users_analysed': 2
    }
    response = metrics.get_metrics()
    response.pop('analysis_completion_date')
    assert response == expected


def test_get_best_params():
    metrics = [None, None, None]
    metrics[0] = {
        'metrics_results': [{
            'name': 'nDCG',
            'version': '1.0',
            'values': {'mean': 0.5}
        }],
        'model_params': 'param_set_a'
    }

    metrics[1] = {
        'metrics_results': [{
            'name': 'nDCG',
            'version': '1.0',
            'values': {'mean': 0.2}
        }],
        'model_params': 'param_set_b'
    }

    metrics[2] = {
        'metrics_results': [{
            'name': 'nDCG',
            'version': '1.0',
            'values': {'mean': 0.6}
        }],
        'model_params': 'param_set_c'
    }

    best_metric = get_best_params(metrics)

    assert best_metric == metrics[2]
