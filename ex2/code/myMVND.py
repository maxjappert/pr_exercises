import math

import numpy as np
from typing import List
from scipy.stats import multivariate_normal


class MVND:
    # TODO: EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean = np.mean(data)
        self.cov = np.cov(data)

    # TODO: EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> np.ndarray:       # Alternatively a float can also be returned if individual datapoints are computed

        d, n = x.shape

        if n is 1:
            #print("single datapoint")
            return multivariate_normal.pdf(x, self.mean, self.cov)

        probabilities = np.array((1, n))

        for i in range(0, n):
            probabilities[n] = multivariate_normal.pdf(x[:i])
            assert(x[:i].shape == (d, 1))

        return probabilities



def log_likelihood(data: np.ndarray, mvnd: List[
    MVND]) -> np.ndarray:  # Alternatively a float can also be returned if individual datapoints are computed
    '''
    Compute the log likelihood of each datapoint
    :param data:    Training inputs, #(samples) x #(dim)
    :param mvnd:     List of MVND objects
    :return:        Likelihood of each data point
    '''
    log_likelihood = np.zeros((1, data.shape[0]))

    # TODO: EXERCISE 2 - Compute likelihood of data
    # Note: For MVGD there will only be 1 item in the list
    for i in range(0, data.shape[0]):
        log_likelihood[1, i] = math.log(mvnd[0].pdf(data[i:]), math.e)

    return log_likelihood
