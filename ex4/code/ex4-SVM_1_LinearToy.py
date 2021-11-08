import matplotlib
import sys
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from svm import SVM
from plotting_helper import plot_data, plot_linear_separator

dataPath = '../data/'


def svmLinearToyExample() -> None:
    '''
     - Load linear separable toy dataset
     - Train a linear SVM
     - Print training and test error
     - Plot data and separator
    '''
    C = None

    toy = scipy.io.loadmat(dataPath + 'toy.mat')
    toy_train = toy['toy_train']
    toy_test = toy['toy_test']

    toy_train_label = np.transpose(toy_train[0, :].astype(np.double)[:, None])
    toy_train_x = toy_train[1:3, :].astype(np.double)

    toy_test_label = np.transpose(toy_test[0, :].astype(np.double)[:, None])
    toy_test_x = toy_test[1:3, :].astype(np.double)

    svm = SVM(C)
    svm.train(toy_train_x, toy_train_label)

    print("Training error")
    svm.printLinearClassificationError(toy_train_x, toy_train_label)

    print("Test error")
    svm.printLinearClassificationError(toy_test_x, toy_test_label)

    print("Visualizing data")
    datamin = math.floor(min(np.min(toy_train_x), np.min(np.max(toy_test_x))))
    datamax = math.ceil(max(np.max(toy_train_x), np.max(np.max(toy_test_x))))

    plot_linear_separator(plt, svm, datamin, datamax)
    plot_data(plt, toy_train_x, toy_train_label, [['red', '+'], ['blue', '_']], 'Training')
    plot_data(plt, toy_test_x, toy_test_label, [['yellow', '+'], ['green', '_']], 'Test')
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Toy Example")
    print("##########-##########-##########")
    svmLinearToyExample()
    print("##########-##########-##########")
