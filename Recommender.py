from collections import Counter
from collections import deque
from pathlib import Path

import implicit
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from utils import *


class SparseRecommender:
    def __init__(self, files, users, d=50, eps=0.1, alpha=1, lam=100, iters=30):
        self.d = d  # embedding size
        self.eps = eps  # data scale
        self.alpha = alpha  # data scale
        self.lam = lam  # regularization
        self.iters = iters # als iterations

        # developers and files containers
        self.files = files
        self.users = users

        # interaction matrices
        self.M_com_raw = csr_matrix((len(self.users), len(self.files)))
        self.M_rev_raw = csr_matrix((len(self.users), len(self.files)))

        # scaled matrices
        self.M_com = csr_matrix((len(self.users), len(self.files)))
        self.M_rev = csr_matrix((len(self.users), len(self.files)))

        # reviewer and developer recommenders
        self.als_rev = None
        self.als_com = None

        self.review_flag = False

    def get_closest(self, target):
        """
        TODO move to utils
        """
        parts = [Path(path).parts for path in self.files.id2obj[self.files.contain_mask != 0]]
        parts_target = Path(self.files.id2obj[target]).parts

        # longest common, longest prefix, index
        longest_common_tokens = deque([(MISSING_ID, MISSING_ID, MISSING_ID)], maxlen=2)
        for ind, _parts in enumerate(parts):
            len_common_tokens = len(Counter(_parts) & Counter(parts_target))
            prefix = []
            for p1, p2 in zip(parts_target, _parts):
                if p1 != p2:
                    break
                prefix.append(p1)
            len_common_prefix = len(prefix)

            if len_common_tokens > 0 and len_common_tokens >= longest_common_tokens[0][0] \
                    and len_common_prefix >= longest_common_tokens[0][1]:
                longest_common_tokens.appendleft((len_common_tokens, len_common_prefix, ind))
        return [self.files.obj2id[self.files.id2obj[ind]] for _, _, ind in longest_common_tokens if ind != MISSING_ID]

    def process_event(self, event):
        if event.type == 'commit':
            authors = np.array([event.author_login])
            files = event.file_path
        else:
            authors = np.hstack((event.author_login, event.participant_login))
            files = event.file_path

        files_to_add = files
        files_rec_ids = []

        for file in files:
            if event.type == 'review' and not self.files.contain(file):
                file_id = self.get_closest(file)
                if len(file_id) > 0:
                    files_rec_ids.append(file_id[0])
            else:
                files_rec_ids.append(file)

        self.users.contain_mask[authors] = 1

        if event.type == 'commit':
            return authors, files_to_add

        return authors, files_to_add, files_rec_ids

    def initialize(self, events):
        return self.process(events, recommend=False)

    def add_files(self, updates):
        for update in updates:
            self.files.contain_mask[update[1]] = 1

    def update(self, updates, matrix):
        for update in updates:
            if len(update[1]) == 0 or len(update[0]) == 0:
                return

            ind_users = update[0].repeat(len(update[1]))
            ind_files = np.tile(update[1], len(update[0]))
            if matrix == 'review':
                self.review_flag = True
                self.M_rev_raw[ind_users, ind_files] += 1
            elif matrix == 'commit':
                self.M_com_raw[ind_users, ind_files] += 1
            else:
                raise ValueError(f'Wrong matrix={matrix} to update')

    def process(self, events, recommend=True):
        review_update = []
        commit_update = []

        preds = []
        target = []

        if recommend:
            sim_uu = cosine_similarity(self.als_com.user_factors)
            sim_uf = self.als_rev.user_factors @ self.als_rev.item_factors.T  # won't work with more complex models

        for i, event in events.iterrows():
            if event.type == 'commit':
                commit_update.append(self.process_event(event))
            else:
                user_ids, files_to_add, files_rec_ids = self.process_event(event)
                if recommend:
                    recs = self.recommend(sim_uu, sim_uf, user_ids, files_rec_ids)
                    preds.append(recs)
                    target.append(event.reviewer_login)
                review_update.append((event.reviewer_login, files_to_add))

        self.add_files(commit_update)
        self.add_files(review_update)

        self.update(review_update, 'review')
        self.update(commit_update, 'commit')

        self.M_com[self.M_com_raw.nonzero()] = self.alpha * np.log(
            self.M_com_raw[self.M_com_raw.nonzero()] / self.eps + 1)
        if self.review_flag:
            self.M_rev[self.M_rev_raw.nonzero()] = self.alpha * np.log(
                self.M_rev_raw[self.M_rev_raw.nonzero()] / self.eps + 1)

        self.als_rev = implicit.als.AlternatingLeastSquares(self.d, iterations=self.iters, regularization=self.lam)
        self.als_com = implicit.als.AlternatingLeastSquares(self.d, iterations=self.iters, regularization=self.lam)

        self.als_rev.fit(self.M_rev.T, show_progress=False)
        self.als_com.fit(self.M_com.T, show_progress=False)

        if recommend:
            return preds, target

    def recommend(self, user_user_similarity, user_file_similarity, user_ids, file_ids):
        """
        TODO move to utils or move to the parent class
        :param user_user_similarity:
        :param user_file_similarity:
        :param user_ids:
        :param file_ids:
        :return:
        """

        reviewer_score = user_file_similarity[:, file_ids]  # sort here and leave only top 10
        developer_score = user_user_similarity @ reviewer_score / (
                user_user_similarity.sum(axis=1).reshape(-1, 1) + 1e-15)

        reviewer_score = reviewer_score.sum(axis=1)
        developer_score = developer_score.sum(axis=1)

        scores = reviewer_score.copy()
        scores[scores == 0] = developer_score[scores == 0]

        if len(user_ids):
            scores[user_ids] = -np.inf

        pred = np.argsort(scores)[-10:]

        if len(pred) < 10:
            fill_pred = -np.ones(10)
            fill_pred[:len(pred)] = pred
            pred = fill_pred

        return pred
