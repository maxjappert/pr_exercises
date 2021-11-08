import sys
import scipy.io
from svm_image_helper import svm_image

dataPath = '../data/'


def testMNIST38() -> None:
    '''
     - Load MNIST dataset, characters 3 and 8
     - Train a kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running MNIST38")
    data = scipy.io.loadmat(dataPath + 'zip38.mat')
    train = data['zip38_train']
    test = data['zip38_test']

    # TODO: Set parameters:
    # linear: boolean (if linear is true, then the kernel parameters are not used)
    # C: float
    # Kernels: 'linear', 'poly', 'rbf'
    # Kernelpar: float
    svm_image(train, test, is_cifar=False, linear=False, C=10, kernel='rbf', kernelpar=0.5)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("SVM exercise - MNIST Example 3 vs 8")
    print("##########-##########-##########")
    testMNIST38()
    print("\n##########-##########-##########")
