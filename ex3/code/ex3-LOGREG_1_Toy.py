import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from logreg import LOGREG
from logregHelper import plot2D, plot3D

dataPath = '../data/'


def logregToy() -> None:
    # load data
    toy = scipy.io.loadmat(dataPath + 'toy.mat')
    toy_train = toy['toy_train']
    toy_test = toy['toy_test']

    train_label = np.transpose(toy_train[0, :].astype(np.double))
    train_label[train_label < 0] = 0.0
    train_x = toy_train[0:3, :].astype(np.double)
    train_x[0, :] = 1.0  # adding row of 1s in X matrix to account for w0 term

    test_label = np.transpose(toy_test[0, :].astype(np.double))
    test_label[test_label < 0] = 0.0
    test_x = toy_test[0:3, :].astype(np.double)
    test_x[0, :] = 1.0  # adding row of 1s in X matrix to account for w0 term

    # training coefficients
    regularization_coefficients = [0.0, 0.1, 1.0]
    # without regularization : regularization_coefficients = 0
    # with regularization    : regularization_coefficients = 1 / 2*sigma^2

    for r in regularization_coefficients:
        print('with regularization coefficient ', r)
        logreg = LOGREG(r)
        trained_w = logreg.train(train_x, train_label, 50)
        print("Training:")
        logreg.printClassification(train_x, train_label)
        print("Test:")
        logreg.printClassification(test_x, test_label)
        print("PARS: {}".format(trained_w))

        # plot for toy dataset
        figname = 'Toy dataset with r: {}'.format(r)
        fig = plt.figure(figname, figsize=(12, 12))
        plt.tight_layout()
        plt.subplot(221)
        plot2D(plt, train_x, train_label, trained_w, 'Training')
        plt.subplot(222)
        plot2D(plt, test_x, test_label, trained_w, 'Testing')
        plot3D(plt, fig.add_subplot(223, projection='3d'), train_x, train_label, trained_w, 'Training')
        plot3D(plt, fig.add_subplot(224, projection='3d'), test_x, test_label, trained_w, 'Test')
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nLOGREG exercise - Toy Example")
    print("##########-##########-##########")
    logregToy()
    print("##########-##########-##########")
