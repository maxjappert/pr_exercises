import matplotlib
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from svm import SVM
from plotting_helper import plot_data, plot_kernel_separator

dataPath = '../data/'


def svmKernelToyExample() -> None:
    '''
     - Load non-linear separable toy data_set
     - Train a kernel SVM
     - Print training and test error
     - Plot data and separator
    '''
    data = scio.loadmat(dataPath + 'flower.mat')
    # Only take a subset of the training data
    train = np.append(data['train'][:, :100], data['train'][:, 2100:2200], axis=1)
    d, n = np.shape(train)
    train_x = train[:d - 1, :]
    train_label = np.reshape(train[d - 1, :], (1, n)).astype(float)
    train_label[train_label == 0.0] = -1.0

    d, n = np.shape(data['test'])
    test_x = data['test'][:d - 1, :]
    test_label = np.reshape(data['test'][d - 1, :], (1, n)).astype(float)
    test_label[test_label == 0.0] = -1.0

    plot_data(plt, train_x, train_label, [['red', '+'], ['blue', '_']], 'Training')
    plot_data(plt, test_x, test_label, [['yellow', '+'], ['green', '_']], 'Test')
    plt.show()

    print("Train kernel svm")
    C = None
    svm = SVM(C)
    # TODO: Try out different kernels and kernel parameter values
    svm.train(train_x, train_label, kernel='rbf', kernelpar=0.5)

    print("Training error")
    svm.printKernelClassificationError(train_x, train_label)
    print("Test error")
    svm.printKernelClassificationError(test_x, test_label)

    print("Visualizing data")
    datamin = math.floor(min(np.min(train_x), np.min(np.max(test_x))))
    datamax = math.ceil(max(np.max(train_x), np.max(np.max(test_x))))

    plot_kernel_separator(plt, svm, datamin, datamax * 1.1, h=0.05)
    plot_data(plt, train_x, train_label, [['red', '+'], ['blue', '_']], 'Training')
    plot_data(plt, test_x, test_label, [['yellow', '+'], ['green', '_']], 'Test')
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Non-linear Toy Example")
    print("##########-##########-##########")
    svmKernelToyExample()
    print("##########-##########-##########")
