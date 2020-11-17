###Optimizer: finds the weights to use with an ensemble of models##

import numpy as np
from functools import partial
from scipy.optimize import fmin
from sklearn import metrics

class OptimizerACC:
    """Class for optimizing Accuracy"""

    def __init__(self):
        self.coef_ = 0

    def _acc(self, coef, X,y):
        """Calculates and return accuracy"""
        #Create predictions multiplying X tiems coefficients
        x_coef = X * coef

        #Create predictions by taking the sume of rows
        predictions = np.sum(x_coef, axis = 1)

        #Calculate accuracy score
        acc_score = metrics.accuracy_score(y, predictions)

    def fit(self, X,y):
        loss_partial = partial(self._acc, X = X, y = y)
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size = 1)
        self.coef_ = fmin(loss_partial, initial_coef, disp = True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis =1)
        return predictions