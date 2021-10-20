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
    # data = data.T
    [d, N] = data.shape
    gmm = []
    for _ in range(0, K):
        gmm.append(MVND)
    # TODO: EXERCISE 2 - Implement E and M step of GMM algorithm
    # Hint - first randomly assign a cluster to each sample
    # Hint - then iteratively update mean, cov and c value of each cluster via EM
    # Hint - use the gmm_draw() function to visualize each step

    cluster_1 = 0
    cluster_2 = 0
    cluster_3 = 0
    cluster_assignments = np.random.randint(0, K, N)

    for i in range(N):
        if cluster_assignments[i] == 0:
            cluster_1 += 1
        elif cluster_assignments[i] == 1:
            cluster_2 += 1
        else:
            cluster_3 += 1
    cluster_one = np.zeros((d, cluster_1))
    cluster_two = np.zeros((d, cluster_2))
    cluster_three = np.zeros((d, cluster_3))

    helper_1 = 0
    helper_2 = 0
    helper_3 = 0
    for i in range(N):
        if cluster_assignments[i] == 0:
            cluster_one[:, helper_1] = data[:, i]
            helper_1 += 1
        elif cluster_assignments[i] == 1:
            cluster_two[:, helper_2] = data[:, i]
            helper_2 += 1
        else:
            cluster_three[:, helper_3] = data[:, i]
            helper_3 += 1

    cs = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    mvnd_1 = MVND(cluster_one, cs[0])
    cov_1 = mvnd_1.cov
    mean_1 = mvnd_1.mean

    mvnd_2 = MVND(cluster_two, cs[1])
    cov_2 = mvnd_2.cov
    mean_2 = mvnd_2.mean

    mvnd_3 = MVND(cluster_three, cs[2])
    cov_3 = mvnd_3.cov
    mean_3 = mvnd_3.mean

    mvnds = [mvnd_1, mvnd_2, mvnd_3]
    covs = [cov_1, cov_2, cov_3]
    means = [mean_1, mean_2, mean_3]

    probs = np.zeros((K, N))

    log_likelihoods = np.zeros((K, N))
    for _ in range(0, iter):
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

        new_log_likelihoods = log_likelihood(data, mvnds)

        difference = abs(new_log_likelihoods - log_likelihoods)
        sum_of_difference = np.sum(difference)

        if sum_of_difference < eps:
            break

        log_likelihoods = log_likelihood(data, mvnds)

    gmm_draw(mvnds, data, 'pls work')
    plt.show()

    gmm[0] = MVND
    gmm[1] = MVND
    gmm[2] = MVND

    gmm[0].cov = covs[0]
    gmm[0].mean = means[0]
    gmm[0].c = cs[0]

    gmm[1].cov = covs[1]
    gmm[1].mean = means[1]
    gmm[1].c = cs[1]

    gmm[2].cov = covs[2]
    gmm[2].mean = means[2]
    gmm[2].c = cs[2]

    return gmm
