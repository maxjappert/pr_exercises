import sys, math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from matplotlib.patches import Ellipse
from myMVND import *


def gmm_draw(gmm, data, plotname='') -> None:
    '''
    gmm helper function to visualize cluster assignment of data
    :param gmm:         list of MVND objects
    :param data:        Training inputs, #(dims) x #(samples)
    :param plotname:    Optional figure name
    '''
    plt.figure(plotname)
    K = len(gmm)
    N = data.shape[1]
    dists = np.zeros((K, N))
    for k in range(0, K):
        d = data - (np.kron(np.ones((N, 1)), gmm[k].mean)).T
        dists[k, :] = np.sum(np.multiply(np.matmul(inv(gmm[k].cov), d), d), axis=0)
    comp = np.argmin(dists, axis=0)

    # plot the input data
    ax = plt.gca()
    ax.axis('equal')
    for (k, g) in enumerate(gmm):
        indexes = np.where(comp == k)[0]
        kdata = data[:, indexes]
        g.data = kdata
        ax.scatter(kdata[0, :], kdata[1, :])

        [_, L, V] = scipy.linalg.svd(g.cov, full_matrices=False)
        phi = math.acos(V[0, 0])
        if float(V[1, 0]) < 0.0:
            phi = 2 * math.pi - phi
        phi = 360 - (phi * 180 / math.pi)
        center = np.array(g.mean).reshape(1, -1)

        d1 = 2 * np.sqrt(L[0])
        d2 = 2 * np.sqrt(L[1])
        ax.add_patch(Ellipse(center.T, d1, d2, phi, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1, fill=False))
        plt.plot(center[0, 0], center[0, 1], 'kx')


def gmm_em(data, K: int, iter: int, plot=False) -> list:
    '''
    EM-algorithm for Gaussian Mixture Models
    Usage: gmm = gmm_em(data, K, iter)
    :param data:    Training inputs, #(dims) x #(samples)
    :param K:       Number of GMM components, integer (>=1)
    :param iter:    Number of iterations, integer (>=0)
    :param plot:    Enable/disable debugging plotting
    :return:        List of objects holding the GMM parameters.
                    Use gmm[i].mean, gmm[i].cov, gmm[i].c
    '''

    eps = sys.float_info.epsilon
    [d, N] = data.shape
    gmm = []
    for _ in range(0, K):
        gmm.append(MVND)
    # EXERCISE 2 - Implement E and M step of GMM algorithm
    # Hint - first randomly assign a cluster to each sample
    # Hint - then iteratively update mean, cov and c value of each cluster via EM
    # Hint - use the gmm_draw() function to visualize each step

    # This is just initialization, which consists of randomly assigning each datapoint
    # to one of K clusters. Each cluster is represented by an MVND object, which
    # again consists of a mean vector, a covariance matrix and a weight c.
    clusters = []
    indexes = []

    assignment = []
    class_sizes = []

    for _ in range(0, K):
        class_sizes.append(0)
        indexes.append(0)

    for i in range(0, N):
        k = np.random.randint(0, K)
        assignment.append(k)
        class_sizes[k] += 1

    for k in range(0, K):
        clusters.append(np.zeros((d, class_sizes[k])))

    for i in range(0, N):
        k = assignment[i]
        clusters[k][:, indexes[k]] = data[:, i]
        indexes[k] += 1

    mvnds = []
    covs = []
    means = []
    cs = []

    for k in range(0, K):
        mvnds.append(MVND(clusters[k], 1.0 / K))
        covs.append(np.ndarray)
        means.append(np.ndarray)
        cs.append(float)

    probs = np.zeros((K, N))

    log_likelihoods = np.zeros((K, N))

    # Here we iteratively perform the E- and M-steps as demonstrated on slide 27.
    # The formulas are copied from there.
    for step in range(0, iter):
        # E-step:
        for i in range(0, N):
            total_probs = 0
            for j in range(0, K):
                total_probs += mvnds[j].c * mvnds[j].pdf(data[:, i])
            for k in range(0, K):
                probs[k, i] = mvnds[k].c * mvnds[k].pdf(data[:, i]) / total_probs

        # M-step:
        cs = [0, 0, 0]
        for k in range(0, K):
            a = 0
            b = 0
            for j in range(0, N):
                cs[k] += probs[k, j]
                a += probs[k, j] * data[:, j]
                b += probs[k, j]
            cs[k] *= (1.0 / N)
            means[k] = a / b

        for k in range(0, K):
            a = 0
            b = 0
            for i in range(0, N):
                x_i = data[:, i]
                a += probs[k, i] * np.matmul(np.matrix(x_i - means[k]).T, np.matrix(x_i - means[k]))
                b += probs[k, i]
            covs[k] = a / b

        for k in range(0, K):
            mvnds[k].cov = covs[k]
            mvnds[k].mean = means[k]
            mvnds[k].c = cs[k]

        # Here we check the difference between the log_likelihood of the previous and the current iterations
        # If the absolute difference is smaller or equal to the system epsilon (so if the difference is de facto
        # inexistent, then we assume that the algorithm has converged and we quit the loop.
        new_log_likelihoods = log_likelihood(data, mvnds)

        difference = abs(new_log_likelihoods - log_likelihoods)
        sum_of_difference = np.sum(difference)

        if sum_of_difference <= eps:
            break

        log_likelihoods = log_likelihood(data, mvnds)

        if plot:
            gmm_draw(mvnds, data, str(step))
            plt.show()

    gmm_draw(mvnds, data, 'after')
    plt.show()

    return mvnds
