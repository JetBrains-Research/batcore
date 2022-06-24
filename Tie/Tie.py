from collections import defaultdict
from datetime import timedelta
from itertools import chain

from RecommenderBase.recommender import RecommenderBase
from Tie.SimilarityRecommender import SimilarityRecommender
from Tie.TextMiningRecommender import TextMiningRecommender


class Tie(RecommenderBase):
    def __init__(self, dataset, alpha=0.5, max_date=100):
        super().__init__()

        self.text_rec = TextMiningRecommender(dataset)
        self.sim_rec = SimilarityRecommender()
        self.alpha = alpha
        self.max_date = timedelta(max_date, 100)

    def predict(self, data, n=10):
        res = []
        for i, row in data.iterrows():
            scores_text = self.text_rec.predict_single_review(row)
            scores_sim = self.sim_rec.predict_single_review(row)

            den_text = sum(scores_text.values())
            den_sim = sum(scores_sim.values())

            if den_text == 0:
                den_text = 1
            if den_sim == 0:
                den_sim = 1

            score = defaultdict(lambda: 0)
            for rev in chain(scores_text.keys(), scores_sim.keys()):
                if rev in scores_text:
                    score[rev] += self.alpha * scores_text[rev] / den_text
                if rev in scores_sim:
                    score[rev] += (1 - self.alpha) * scores_sim[rev] / den_sim
            best = [k for k, v in sorted(score.items(), key=lambda item: -item[1])]
            best = best[:n]
            res.append(best)

        return res

    def fit(self, data):
        self.text_rec.fit(data)
        self.text_rec.set_new_ids()
        self.sim_rec.fit(data)