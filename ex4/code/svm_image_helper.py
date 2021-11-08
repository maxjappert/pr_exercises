import numpy as np
from svm import SVM
import matplotlib.pyplot as plt
from plotting_helper import visualizeClassification


def svm_image(train: np.ndarray, test: np.ndarray, is_cifar: bool, linear: bool = True, C:float = 10.0, kernel: str = 'rbf',
              kernelpar: float = 0.5) -> SVM:
    '''
    Train an SVM with the given training data and print training + test error
    :param train: Training data
    :param test: Test data
    :param is_cifar: using the cifar (true) dataset or mnist (false)
    :return: Trained SVM object
    '''

    _, N = np.shape(train)
    if is_cifar:
        # Adapt to the size of the training set to gain speed or precision
        N = 200

    train_label = np.reshape(train[0, :N], (1, N)).astype(float)
    train_label[train_label == 0] = -1.0
    train_x = train[1:, :N].astype(float)

    _, n = np.shape(test)
    test_label = np.reshape(test[0, :], (1, n)).astype(float)
    test_label[test_label == 0] = -1.0
    test_x = test[1:, :].astype(float)


    svm = SVM(C)
    if linear:
        svm.train(train_x, train_label)
    else:
        svm.train(train_x, train_label, kernel=kernel, kernelpar=kernelpar)

    print("Training error")
    if linear:
        svm.printLinearClassificationError(train_x, train_label)
    else:
        svm.printKernelClassificationError(train_x, train_label)

    print("Test error")
    if linear:
        svm.printLinearClassificationError(test_x, test_label)
    else:
        svm.printKernelClassificationError(test_x, test_label)
    plt.show()

    if linear:
        visualizeClassification(train_x, train_label, svm.classifyLinear(train_x), 3 * 3, is_cifar, 'training')
        visualizeClassification(test_x, test_label, svm.classifyLinear(test_x), 3 * 3, is_cifar, 'test')
    else:
        visualizeClassification(train_x, train_label, svm.classifyKernel(train_x), 3 * 3, is_cifar, 'training')
        visualizeClassification(test_x, test_label, svm.classifyKernel(test_x), 3 * 3, is_cifar, 'test')
    return svm
