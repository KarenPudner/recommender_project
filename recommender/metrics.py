import os
import numpy as np
from datetime import datetime
import json
from recommender import ndcg


class Metrics:
    def __init__(self):

        self.num_users = 0
        self.sum_user_ndcg = 0

    def update_all_metrics(self, groundtruth, rec_list):
        """ Receives a list of groundtruth item IDs and list of recommendations and updates all the metrics that are
        implemented in this file.

        Currently implemented metrics in this file:
            * NDCG

        Args:
            groundtruth (list): list of consumed item IDs by current user
            rec_list (list): list of recommended item IDs
        Returns:
            None
        """
        # Only update the metrics object if the rec_list is populated.
        if len(rec_list) > 0:
            # Update metrics
            self.update_ndcg(groundtruth, rec_list)

    def get_current_ndcg(self):
        """
        Calculates and returns the mean ndcg over all users calculated in this object.
        """
        if self.num_users != 0:
            return self.sum_user_ndcg / self.num_users
        else:
            return 0

    @staticmethod
    def calculate_ndcg(groundtruth, rec_list, k=20):
        relevance_scores = np.zeros(len(rec_list)).tolist()

        for i, item in enumerate(rec_list):
            if item in groundtruth:
                relevance_scores[i] = 1

        score = ndcg.ndcg_at_k(relevance_scores, k, 1)

        return score

    def update_ndcg(self, groundtruth, rec_list):
        score = self.calculate_ndcg(groundtruth, rec_list, 20)
        self.sum_user_ndcg += score
        self.num_users += 1

    def get_metrics(self):
        """
        Uses relevant object variables to calculate final metrics and returns them.
        """
        # Calculate metrics
        values = {}
        values["mean"] = self.get_current_ndcg()

        ndcg_metric = {}
        ndcg_metric["name"] = "nDCG"
        ndcg_metric["version"] = "1.0"
        ndcg_metric["values"] = values

        metrics_results = []
        metrics_results.append(ndcg_metric)

        # Package metrics up into single dictionary
        results = {}
        results["number_of_users_analysed"] = self.num_users
        results["analysis_completion_date"] = str(datetime.now())
        results["metrics_results"] = metrics_results

        return results

    def save_results(self, file_name=None):
        """
         Gets final metrics then saves the to a local JSON file.
        """
        results = self.get_metrics()

        if not file_name:
            file_name = os.path.join(os.path.dirname(__file__), '..', 'results',
                                     'metrics_d{}.json'.format(str(datetime.now())))

        with open(file_name, 'w') as fp:
            json.dump(results, fp)


def get_best_params(metric_results):
    best_result = metric_results[0]

    for current_metric in metric_results[1:]:
        ndcg_current = [i for i in current_metric["metrics_results"] if i["name"] is "nDCG"][0]
        ndcg_best = [i for i in best_result["metrics_results"] if i["name"] is "nDCG"][0]
        if ndcg_current['values']['mean'] > ndcg_best['values']['mean']:
            best_result = current_metric

    return best_result
