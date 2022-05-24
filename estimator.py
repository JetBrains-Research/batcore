import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator


class ALSEstimator(BaseEstimator, AlternatingLeastSquares):
    def __init__(self,
                 factors=100,
                 regularization=.01,
                 iterations=15,
                 filter_already_used=True,
                 calculate_training_loss=True):

        super().__init__(factors=factors, regularization=regularization, iterations=iterations,
                         calculate_training_loss=calculate_training_loss)

        self.x_train_nonzero = None
        self.filter_already_used = filter_already_used

    def fit(self, X, y=None):
        # print(X)
        super().fit(X, show_progress=False)

        if self.filter_already_used:
            self.x_train_nonzero = X.nonzero()

        return self

    def predict(self, X=None, y=None):
        predictions = np.dot(self.item_factors, self.user_factors.T)

        if self.filter_already_used:
            predictions[self.x_train_nonzero] = -99

        return predictions


def train_als(x_df, alpha=30, eps=0.01, factors=50, iterations=30, regularization=100, shape=None):
    row = x_df.login
    col = x_df.file_path
    val = x_df.number

    val = 1 + alpha * np.log(1 + val / eps)
    if shape is None:
        mat_df = coo_matrix((val, (row, col)), ).tocsr()
    else:
        mat_df = coo_matrix((val, (row, col)), shape=shape).tocsr()

    als = ALSEstimator(factors=factors,
                       iterations=iterations,
                       regularization=regularization)

    als.fit(mat_df)

    return als, mat_df