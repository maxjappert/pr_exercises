import sys
import scipy.io
from svm_image_helper import svm_image

dataPath = '../data/'


def testMNIST13() -> None:
    '''
     - Load MNIST dataset, characters 3 and 8
     - Train a kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST13")
    data = scipy.io.loadmat(dataPath + 'zip13.mat')
    train = data['zip13_train']
    test = data['zip13_test']

    # TODO: Set parameters:
    # linear: boolean (if linear is true, then the kernel parameters are not used)
    # C: float
    # Kernels: 'linear', 'poly', 'rbf'
    # Kernelpar: float
    svm_image(train, test, is_cifar=False, linear=False, C=10, kernel='rbf', kernelpar=0.5)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("SVM exercise - MNIST Example 1 vs 3")
    print("##########-##########-##########")
    testMNIST13()
    print("\n##########-##########-##########")
