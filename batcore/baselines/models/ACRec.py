from collections import defaultdict
from datetime import timedelta

from batcore.modelbase.recommender import BanRecommenderBase


class ACRec(BanRecommenderBase):
    """
    ACRec recommends reviewers based on how much they commented on recent pull requests. For this ACRec looks on
    previous reviews in a given timeframe and for each comment assigns its commenter a score based on the time
    passed. Candidates with the best accumulated scores are suggested as reviewers

    Paper: `Who Should Comment on This Pull Request? Analyzing Attributes for More Accurate Commenter Recommendation in Pull-Based Development <https://www.sciencedirect.com/science/article/abs/pii/S095058491630283X?via%3Dihub>`_

    :param gamma: number of days to pass for a pull request to ignored during predictions
    :param lambd: time-decaying parameter
    :param no_owner: flag to add or remove owners of the pull request from the recommendations
    :param no_inactive: flag to add or remove inactive reviewers from recommendations
    :param inactive_time: number of consecutive days without any actions needed to be considered an inactive
    """

    def __init__(self,
                 gamma=60,
                 lambd=0.5,
                 no_owner=True,
                 no_inactive=True,
                 inactive_time=60):
        super().__init__(no_owner, no_inactive, inactive_time)

        self.gamma = timedelta(days=gamma)
        self.lambd = lambd

        self.history = []
        self.commenters = defaultdict(lambda: [])

    def predict(self, pull, n=10):
        """
        goes through recent pull requests and accumulates score for each commenter based on the recency of their comment

        :param pull: pull requests for which reviwers are required
        :param n: number of reviewers to recommend
        :return: at most n reviewers for the pull request
        """
        scores = defaultdict(lambda: 0)
        date = pull['date']
        for review_old in reversed(self.history):
            if (date - review_old['date']) > self.gamma:
                break

            for com in self.commenters[review_old['key_change']]:
                scores[com] += pow((date - review_old['date']).days + 0.01, -self.lambd)

        self.filter(scores, pull)
        sorted_users = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_users[:n]

    def fit(self, data):
        """
        remembers each pull request and build relation between them and comments

        :param data: a batch of pull requests and comments
        """
        super().fit(data)
        for event in data:
            if event['type'] == 'pull':
                self.history.append(event)
            elif event['type'] == 'comment':
                self.commenters[event['key_change']].append(event['key_user'])
