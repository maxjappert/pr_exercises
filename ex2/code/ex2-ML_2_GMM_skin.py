import sys, os
import scipy.io
import matplotlib.pyplot as plt

from imageHelper import imageHelper
from classifyHelper import classify
from imagePrior import get_prior
from myGMM import gmm_em

dataPath = '../data/'


def gmmSkinDetection() -> None:
    '''
    Skin detection - train a GMM for both skin and non-skin.
    Classify the test and training image using the classify helper function.
    Note that the "mask" binary images are used as the ground truth.
    '''
    K = 3
    iter = 50
    sdata = scipy.io.loadmat(os.path.join(dataPath, 'skin.mat'))['sdata']
    ndata = scipy.io.loadmat(os.path.join(dataPath, 'nonskin.mat'))['ndata']
    gmms = gmm_em(sdata, K, iter)
    gmmn = gmm_em(ndata, K, iter)

    print("TRAINING DATA")
    trainingmaskObj = imageHelper()
    trainingmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    trainingimageObj = imageHelper()
    trainingimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))
    prior_skin, prior_nonskin = get_prior(trainingmaskObj)
    classify(trainingimageObj, trainingmaskObj, gmms, gmmn, "Training-GMM", prior_skin=prior_skin,
             prior_nonskin=prior_nonskin)

    print("TEST DATA PORTRAIT")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test1.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test1.png'))
    classify(testimageObj, testmaskObj, gmms, gmmn, "Test-portrait-GMM", prior_skin=prior_skin,
             prior_nonskin=prior_nonskin)

    print("TEST DATA FAMILY")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test2.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test2.png'))
    classify(testimageObj, testmaskObj, gmms, gmmn, "Test-family-GMM", prior_skin=prior_skin,
             prior_nonskin=prior_nonskin)

    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nGMM exercise - Skin detection")
    print("##########-##########-##########")
    gmmSkinDetection()
    print("##########-##########-##########")
