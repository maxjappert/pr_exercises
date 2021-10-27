import sys
import scipy.io
from logregHelper import logreg_image

dataPath = '../data/'


def test_plane() -> None:
    '''
     - Load CIFAR dataset, classes plane and no_plane
     - Train a logistic regression classifier
     - Print training and test error
     - Visualize randomly chosen misclassified and correctly classified
    '''
    print("Running Plane vs no-plane")
    data = scipy.io.loadmat(dataPath + 'plane_no_plane.mat')
    train = data['train']
    test = data['test']
    logreg_image(train, test, regularization_coefficients=[0.0, 0.1, 1.0], is_cifar=True)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\n##########-##########-##########")
    print("LOGREG exercise - CIFAR Example")
    print("##########-##########-##########")
    test_plane()
