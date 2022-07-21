# based on https://github.com/dezhen-k/icsme2015-paper-impl

from datetime import datetime, timedelta

from RecommenderBase.recommender import RecommenderBase
from Tie.utils import get_map


class Tie(RecommenderBase):
    def __init__(self, word_list, reviewer_list, text_splitter=lambda x: x.split(' '), alpha=0.7, max_date=100):
        """
        :param word_list: word dictionary for pulls comments
        :param reviewer_list: list of all reviewers
        :param text_splitter: a function to parse pull comments
        :param alpha: weight between path-based and text-based recommenders
        :param max_date: time in days after which reviews are not considered
        """
        super().__init__()
        self.reviews = []
        self.word_list = word_list
        self.word_map = get_map(word_list)

        self.reviewer_list = reviewer_list
        self.reviewer_map = get_map(reviewer_list)
        self.review_count_map = {}

        self.text_models = [dict() for _ in range(len(reviewer_list))]
        self._similarity_cache = {}
        self.text_splitter = text_splitter

        self.max_date = max_date
        self.alpha = alpha

    def predict(self, review, n=10):
        """Recommends appropriate reviewers of the given review.
            This method returns `max_count` reviewers at most.
        """
        review = self._transform_review_format(review)
        L = []
        for j in range(len(self.reviewer_list)):
            # c = (1 - self.alpha) * self._get_conf_text(review, j) \
            #    + self.alpha * self._get_conf_path(review, j)
            # conf_text = self._get_conf_text(review, j)
            conf_path = self._get_conf_path(review, j)

            # L.append([j, conf_text, 0])
            L.append([j, 0, conf_path])
            # L.append([j, conf_text, conf_path])
        conf_text_sum = sum(map(lambda x: x[1], L))
        conf_path_sum = sum(map(lambda x: x[2], L))
        if conf_text_sum == 0:
            conf_text_sum = 1e-15
        if conf_path_sum == 0:
            conf_path_sum = 1e-15
        for triple in L:
            triple[1] /= conf_text_sum
            triple[2] /= conf_path_sum

        L.sort(key=lambda x: x[1] * self.alpha + x[2] * (1 - self.alpha), reverse=True)
        scores = [x[2] for x in L[:n]]
        L = list(
            map(lambda x: self.reviewer_list[x],
                map(lambda x: x[0], L)
                )
        )
        return L[:n]

    def fit(self, review):
        """Updates the state of the model with an input review."""
        review = self._transform_review_format(review)

        if len(review["body"]) == 0:
            raise Exception("Cannot update.")

        for reviewer_index in review["reviewer_login"]:
            self.review_count_map[reviewer_index] = \
                self.review_count_map.get(reviewer_index, 0) + 1
            for word_index in review["body"]:
                self.text_models[reviewer_index][word_index] = \
                    self.text_models[reviewer_index].get(word_index, 0) + 1

        self.reviews.append(review)

    def _get_conf_path(self, review, reviewer_index):

        s = 0
        end_time = review["date"]
        start_time = (datetime.fromtimestamp(end_time) - timedelta(days=self.max_date)).timestamp()
        for old_rev in reversed(self.reviews):
            if old_rev["date"] == end_time:
                continue

            if old_rev["date"] < start_time:
                break

            c = self._calc_similarity(old_rev, review)
            for index in old_rev["reviewer_login"]:
                if index == reviewer_index:
                    s += c
                    break
        return s

    def _get_conf_text(self, review, reviewer_index):
        product = 1
        s = 0
        for _, v in self.text_models[reviewer_index].items():
            s += v
        for word_index in review["body"]:
            p = self.text_models[reviewer_index].get(word_index, 1e-9) / (s + 1)
            product *= p
        return self.review_count_map.get(reviewer_index, 0) / len(self.reviews) * product

    def _calc_similarity(self, rev1, rev2):
        key = str(rev1["id"]) + "-" + str(rev2["id"])
        if key in self._similarity_cache:
            return self._similarity_cache[key]
        changed_files1 = rev1["file_path"][:500]
        changed_files2 = rev2["file_path"][:500]
        if len(changed_files1) == 0 or len(changed_files2) == 0:
            return 0
        sum_score = 0
        for f1 in changed_files1:
            s1 = set(f1.split('/'))
            for f2 in changed_files2:
                s2 = set(f2.split('/'))
                sum_score += (len(s1 & s2)) / max(len(s1), len(s2))
        ret = sum_score / (len(changed_files1) * len(changed_files2) + 1)
        self._similarity_cache[key] = ret
        return ret

    def _transform_review_format(self, review):
        word_indices = list(map(lambda x: self.word_map[x],
                                filter(lambda x: x in self.word_map.keys(),
                                       self.text_splitter(review["body"])
                                       )
                                ))
        reviewer_indices = [self.reviewer_map[_reviewer] for _reviewer in review["reviewer_login"]]
        return {
            "body": word_indices,
            "reviewer_login": reviewer_indices,
            "id": review["number"],
            "date": review["date"].timestamp(),
            "file_path": review["file_path"]
        }

    def _review_history_start_index(self, t):
        i = 0
        j = len(self.reviews) - 1
        while i < j:
            mid = (i + j) // 2
            if self.reviews[mid]["date"] <= t:
                i = mid + 1
            else:
                j = mid

        if self.reviews[i]["date"] > t:
            return i
        else:
            return -1

    def _review_history_end_index(self, t):
        i = 0
        j = len(self.reviews) - 1
        while i < j:
            mid = (i + j + 1) // 2
            if self.reviews[mid]["date"] < t:
                i = mid
            else:
                j = mid - 1

        if self.reviews[i]["date"] < t:
            return i
        else:
            return -1