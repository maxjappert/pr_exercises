import sys
import scipy.io
from logregHelper import logreg_image

dataPath = '../data/'


def testMNIST38() -> None:
    '''
     - Load MNIST dataset, characters 3 and 8
     - Train a logreg classifier
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    data = scipy.io.loadmat(dataPath + 'zip38.mat')
    train = data['zip38_train']
    test = data['zip38_test']
    logreg_image(train, test, regularization_coefficients=[0.0, 0.1, 1.0])


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("LOGREG exercise - MNIST Example 3 vs 8")
    print("##########-##########-##########")
    testMNIST38()
    print("\n##########-##########-##########")
