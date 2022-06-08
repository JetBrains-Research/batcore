import pandas as pd

from utils import count_metrics


class Tester:
    # TODO add col for date as parameter
    def test_recommender(self,
                         recommender,
                         dataset,
                         initial_delta=2700,
                         test_interval: int = 7,
                         top_ns: list = None):
        if top_ns is None:
            top_ns = [1, 3, 5, 10]

        dataset.set_params(initial_delta, test_interval)
        recs = []
        cnt = 0
        for (train_data, test_data) in dataset:
            cnt += 1
            recommender.fit(train_data)
            if len(test_data):
                cur_rec = recommender.predict(test_data, n=max(top_ns))
                y_pred = []
                for i, row in test_data.reset_index().iterrows():
                    y_pred.append([*[cur_rec[i][:n] for n in top_ns], row.reviewer_login])
                y_pred = pd.DataFrame(y_pred, columns=[*[f'top-{n}' for n in top_ns], 'rev'])
                recs.append(y_pred)
            # if cnt > 150:
            #     break
            # # print(cnt)

        recs = pd.concat(recs, axis=0)

        return count_metrics(recs)
