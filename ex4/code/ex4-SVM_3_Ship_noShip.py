import sys
import scipy.io
from svm_image_helper import svm_image

dataPath = '../data/'


def test_ship() -> None:
    '''
     - Load CIFAR dataset, classes ship and no_ship
     - Train a linear or kernel SVM
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running Ship or no-ship")
    toy = scipy.io.loadmat(dataPath + 'ship_no_ship.mat')
    train = toy['train']
    test = toy['test']

    # TODO: Set parameters:
    # linear: boolean (if linear is true, then the kernel parameters are not used)
    # C: float
    # Kernels: 'linear', 'poly', 'rbf'
    # Kernelpar: float
    svm_image(train, test, is_cifar=True, linear=False, C=10, kernel='rbf', kernelpar=0.5)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("SVM exercise - CIFAR Example Ship vs no Ship")
    print("##########-##########-##########")
    test_ship()
    print("##########-##########-##########")
