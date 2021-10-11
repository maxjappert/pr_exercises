import sys, os
import scipy.io

from myGMM import gmm_em

dataPath = '../data/'


def gmmToyExample() -> None:
    '''
    GMM toy example - load toyexample data and visualize cluster assignment of each datapoint
    '''
    gmmdata = scipy.io.loadmat(os.path.join(dataPath, 'gmmdata.mat'))['gmmdata']
    gmm_em(gmmdata, 3, 20, plot=True)


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nGMM exercise - Toy example")
    print("##########-##########-##########")
    gmmToyExample()
    print("##########-##########-##########")
