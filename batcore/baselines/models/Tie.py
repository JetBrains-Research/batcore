# based on https://github.com/dezhen-k/icsme2015-paper-impl
import copy
from datetime import timedelta

import numpy as np

from batcore.modelbase.recommender import BanRecommenderBase
from ..utils import get_map, pull_sim


class Tie(BanRecommenderBase):
    """
    Tie recommends reviewers based on file paths and the title. Each candidate is assigned two scores. One is based
    on path distance between files in current pr and previously reviewed file. Second is a score from naive Bayes
    classifier trained on the titles of prs.

    Paper: `Who Should Review This Change? Putting Text and File Location Analyses Together for More Accurate Recommendations <https://xin-xia.github.io/publication/icsme15.pdf>`_

    :param item_list: dict with word_list and reviewer_list
    :param text_splitter: a function to parse pull comments
    :param alpha: weight between path-based and text-based recommenders
    :param max_date: time in days after which reviews are not considered
    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 item_list,
                 text_splitter=lambda x: x.split(' '),
                 alpha=0.7,
                 max_date=100,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):

        super().__init__(no_owner, no_inactive, inactive_time)
        self.history = []
        self.word_list = item_list['word_list']
        self.word_map = get_map(item_list['word_list'])

        self.reviewer_list = item_list['reviewer_list']
        self.reviewer_map = get_map(item_list['reviewer_list'])
        self.review_count_map = {}

        self.text_models = [dict() for _ in range(len(item_list['reviewer_list']))]
        self._similarity_cache = {}
        self.text_splitter = text_splitter

        self.max_date = max_date
        self.alpha = alpha

    def predict(self, pull, n=10):
        """
        Recommends appropriate reviewers of the given review. This method returns `max_count` reviewers at most.

        :param n: number of candidates to return
        """
        pull = self.update_pull(pull)

        bayes_scores = [self.bayes_score(pull, j) for j in range(len(self.reviewer_list))]
        fps_scores = self.fps_score(pull)
        L = [[j, bayes_scores[j], fps_scores[j]] for j in range(len(self.reviewer_list))]

        conf_text_sum = sum(map(lambda x: x[1], L))
        conf_path_sum = sum(map(lambda x: x[2], L))
        if conf_text_sum == 0:
            conf_text_sum = 1e-15
        if conf_path_sum == 0:
            conf_path_sum = 1e-15
        for triple in L:
            triple[1] /= conf_text_sum
            triple[2] /= conf_path_sum

        scores = {self.reviewer_list[x[0]]: x[1] * self.alpha + x[2] * (1 - self.alpha) for x in L}
        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_users[:n]

    def fit(self, data):
        """Updates the state of the model with an input review."""
        super().fit(data)
        for event in data:
            if event['type'] == 'pull':
                pull = event
                pull = self.update_pull(pull)

                for reviewer_index in pull["reviewer"]:
                    self.review_count_map[reviewer_index] = \
                        self.review_count_map.get(reviewer_index, 0) + 1
                    for word_index in pull["title"]:
                        self.text_models[reviewer_index][word_index] = \
                            self.text_models[reviewer_index].get(word_index, 0) + 1

                self.history.append(pull)

    def fps_score(self, pull):
        end_time = pull["date"]
        start_time = end_time - timedelta(days=self.max_date)
        scores = np.zeros(len(self.reviewer_list))
        for old_pull in reversed(self.history):
            if old_pull["date"] == end_time:
                continue

            if old_pull["date"] < start_time:
                break

            c = pull_sim(old_pull, pull)
            scores[old_pull['reviewer']] += c

        return scores

    def bayes_score(self, pull, reviewer_index):
        """
        Assigns score to each candidate based on naive bayes classifier trained on pull titles
        """
        product = 1
        s = 0
        for _, v in self.text_models[reviewer_index].items():
            s += v
        for word_index in pull["title"]:
            p = self.text_models[reviewer_index].get(word_index, 1e-9) / (s + 1)
            product *= p
        return self.review_count_map.get(reviewer_index, 0) / (len(self.history) + 1e-8) * product

    def update_pull(self, pull):
        """
        turns title into bag of words vector and replaces reviewers with their ids
        """
        pull = copy.deepcopy(pull)

        word_indices = list(map(lambda x: self.word_map[x],
                                filter(lambda x: x in self.word_map.keys(),
                                       self.text_splitter(pull["title"])
                                       )
                                ))
        reviewer_indices = [self.reviewer_map[_reviewer] for _reviewer in pull["reviewer"]]

        pull['title'] = word_indices
        pull['reviewer'] = reviewer_indices

        return pull
