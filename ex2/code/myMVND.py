import math

import numpy as np
from typing import List
from scipy.stats import multivariate_normal


class MVND:
    # TODO: EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean = np.mean(data, axis=1)
        self.cov = np.cov(data)

    # TODO: EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> np.ndarray:       # Alternatively a float can also be returned if individual datapoints are computed

        return multivariate_normal.pdf(x, self.mean, self.cov)



def log_likelihood(data: np.ndarray, mvnd: List[
    MVND]) -> np.ndarray:  # Alternatively a float can also be returned if individual datapoints are computed
    '''
    Compute the log likelihood of each datapoint
    :param data:    Training inputs, #(samples) x #(dim)
    :param mvnd:     List of MVND objects
    :return:        Likelihood of each data point
    '''

    # switch here!
    d, n = data.shape

    log_likelihood = np.zeros((len(mvnd), n))

    # TODO: EXERCISE 2 - Compute likelihood of data
    # TODO: UPDATE TEMPORARY CHANGE!!!!!
    for j in range(0, len(mvnd)):
        for k in range(0, n):
            log_likelihood[j, k] = math.log(mvnd[j].c * mvnd[j].pdf(data[:, k]), math.e)

    return log_likelihood
