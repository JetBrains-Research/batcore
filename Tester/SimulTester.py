from tqdm import tqdm

from Tester.TesterBase import TesterBase


class SimulTester(TesterBase):
    def __init__(self):
        self.simulated = []
        self.real = []

    def test_recommender(self,
                         recommender,
                         dataset,
                         *args, **kwargs):
        raise NotImplementedError

    def simulate(self, recommender, dataset):
        cnt = 0
        for i, (train_data, test_data) in tqdm(enumerate(dataset)):
            cnt += 1
            if i == 0:
                recommender.fit(train_data)
                self.simulated.append(train_data)
                self.real.append(train_data)

            cur_rec = recommender.predict(test_data, n=1)[0]
            new_train_data = dataset.replace(test_data, cur_rec)

            if i > 0:
                self.simulated.append(new_train_data)
                self.real.append(test_data)
                recommender.fit(new_train_data)

            if cnt > 1000:
                break

    def count_metric_dif(self, metric):
        original_metric = metric(self.real)
        simulated_metric = metric(self.simulated)

        dif = (simulated_metric / original_metric - 1) * 100

        return dif