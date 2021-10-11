import sys, os
import scipy.io
import matplotlib.pyplot as plt

from myMVND import MVND
from imageHelper import imageHelper
from classifyHelper import classify
from imagePrior import get_prior

dataPath = '../data/'


def mvndSkinDetection() -> None:
    '''
    Skin detection - compute a MVND for both skin and non-skin.
    Classify the test and training image using the classify helper function.
    Note that the "mask" binary images are used as the ground truth.
    '''
    sdata = scipy.io.loadmat(os.path.join(dataPath, 'skin.mat'))['sdata']
    ndata = scipy.io.loadmat(os.path.join(dataPath, 'nonskin.mat'))['ndata']

    mvn_sskin = [MVND(sdata)]
    mvn_nskin = [MVND(ndata)]
    # Optain priors
    mask = imageHelper()
    mask.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    # TODO: EXERCISE 2 - Compute the skin and nonskin prior
    prior_skin, prior_nonskin = get_prior(mask)

    print("TRAINING DATA")
    trainingmaskObj = imageHelper()
    trainingmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    trainingimageObj = imageHelper()
    trainingimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))
    classify(trainingimageObj, trainingmaskObj, mvn_sskin, mvn_nskin, "Training-MVND", prior_skin, prior_nonskin)

    print("TEST DATA PORTRAIT")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test1.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test1.png'))
    classify(testimageObj, testmaskObj, mvn_sskin, mvn_nskin, "Test-portrait-MVND", prior_skin, prior_nonskin)

    print("TEST DATA FAMILY")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test2.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test2.png'))
    classify(testimageObj, testmaskObj, mvn_sskin, mvn_nskin, "Test-family-MVND", prior_skin, prior_nonskin)
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nMVND exercise - Skin detection")
    print("##########-##########-##########")
    mvndSkinDetection()
    print("##########-##########-##########")
