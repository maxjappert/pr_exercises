import timeit

import matplotlib
import sys
import time
import numpy as np
import scipy.io

from svm import SVM

dataPath = '../data/'




def svmSpeedComparison() -> None:
    '''
    Train a linear SVM and a Kernel SVM with a linear kernel
    Time the classification functions
    Note the average time over 1000 runs for both classifiers
    '''
    numOfRuns = 1000
    print("Speed comparison")

    toy = scipy.io.loadmat(dataPath + 'toy.mat')
    toy_train = toy['toy_train']
    toy_test = toy['toy_test']

    toy_train_label = np.transpose(toy_train[0, :].astype(np.double)[:, None])
    toy_train_x = toy_train[1:3, :].astype(np.double)

    toy_test_x = toy_test[1:3, :].astype(np.double)


    # we measure the difference between the start- and end-times in seconds
    linear_starttime = time.time()
    for _ in range(0, 1000):
        linear_svm = SVM(None)
        linear_svm.train(toy_train_x, toy_train_label, None, None)
        linear_svm.classifyLinear(toy_test_x)
    linear_endtime = time.time()

    kernel_starttime = time.time()
    for _ in range(0, 1000):
        kernel_svm = SVM(None)
        kernel_svm.train(toy_train_x, toy_train_label, "linear", None)
        kernel_svm.classifyKernel(toy_test_x)
    kernel_endtime = time.time()

    # TODO: Compute the average classification time of both the linear and the kernel SVM (with a linear kernel)
    result_linear = float(linear_endtime - linear_starttime) / 1000
    result_kernel = float(kernel_endtime - kernel_starttime) / 1000

    print('Linear SVM timing: \n {:.10f} over {} runs'.format(result_linear, numOfRuns))
    print('SVM with linear kernel timing: \n {:.10f} over {} runs'.format(result_kernel, numOfRuns))
    print('Linear is {} times faster'.format(result_kernel / result_linear))


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Speed comparison")
    print("##########-##########-##########")
    svmSpeedComparison()
    print("##########-##########-##########")
