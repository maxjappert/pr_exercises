import math
import numpy as np
import matplotlib.pyplot as plt
from imageHelper import imageHelper
from myMVND import MVND, log_likelihood
from typing import List


def classify(img: imageHelper, mask: imageHelper, skin_mvnd: List[MVND], notSkin_mvnd: List[MVND], fig: str = "",
             prior_skin: float = 0.5, prior_nonskin: float = 0.5) -> None:
    '''
    :param img:             imageHelper object containing the image to be classified
    :param mask:            imageHelper object containing the ground truth mask
    :param skin_mvnd:            MVND object for the skin class
    :param notSkin_mvnd:            MVND object for the non-skin class
    :param fig:             Optional figure name
    :param prior_skin:      skin prior, float (0.0-1.0)
    :param prior_nonskin:   nonskin prior, float (0.0-1.0)
    '''
    im_rgb_lin = img.getLinearImage()
    if (type(skin_mvnd) != list):
        skin_mvnd = [skin_mvnd]
    if (type(notSkin_mvnd) != list):
        notSkin_mvnd = [notSkin_mvnd]
    log_likelihood_of_skin_rgb = np.sum(log_likelihood(im_rgb_lin, skin_mvnd), axis=0)
    log_likelihood_of_nonskin_rgb = np.sum(log_likelihood(im_rgb_lin, notSkin_mvnd), axis=0)

    testmask = mask.getLinearImageBinary().astype(int)[:, 0]
    npixels = len(testmask)

    log_likelihood_rgb = log_likelihood_of_skin_rgb - log_likelihood_of_nonskin_rgb
    skin = (log_likelihood_rgb > 0).astype(int)
    print(skin.shape)
    imgMinMask = skin - testmask
    # EXERCISE 2 - Error Rate without prior

    print(imgMinMask.shape)

    fn = 0
    fp = 0

    # The imgMinMask for a given index n is 0 iff the pixel was correctly classified (see the initialization above,
    # where the classification gets compared to the ground truth mask by converting to a binary image and subtracting).
    # if imgMinMask[n] == 1, then that pixel was classified as a false positive, and vice versa with -1.
    for n in range(0, len(imgMinMask)):
        if imgMinMask[n] == 1:
            fp += 1
        elif imgMinMask[n] == -1:
            fn += 1

    totalError = fp + fn

    fn = fn / len(imgMinMask)
    fp = fp / len(imgMinMask)

    print('----- ----- -----')
    print('Total Error WITHOUT Prior =', totalError)
    print('false positive rate =', fp)
    print('false negative rate =', fn)

    #EXERCISE 2 - Error Rate with prior

    log_likelihood_rgb_with_prior = log_likelihood_of_skin_rgb * prior_skin - log_likelihood_of_nonskin_rgb * prior_nonskin
    skin_prior = (log_likelihood_rgb_with_prior > 0).astype(int)
    imgMinMask_prior = skin_prior - testmask
    fp_prior = 0
    fn_prior = 0

    for n in range(0, len(imgMinMask)):
        if imgMinMask_prior[n] == 1:
            fp_prior += 1
        elif imgMinMask_prior[n] == -1:
            fn_prior += 1

    totalError_prior = fp_prior + fn_prior

    fp_prior = fp_prior / len(imgMinMask)
    fn_prior = fn_prior / len(imgMinMask)

    print('----- ----- -----')
    print('Total Error WITH Prior =', totalError_prior)
    print('false positive rate =', fp_prior)
    print('false negative rate =', fn_prior)
    print('----- ----- -----')
    print('skin prior: ', prior_skin)
    print('nonskin prior: ', prior_nonskin)

    N = mask.N
    M = mask.M
    fpImage = np.reshape((imgMinMask > 0).astype(float), (N, M))
    fnImage = np.reshape((imgMinMask < 0).astype(float), (N, M))
    fpImagePrior = np.reshape((imgMinMask_prior > 0).astype(float), (N, M))
    fnImagePrior = np.reshape((imgMinMask_prior < 0).astype(float), (N, M))
    prediction = imageHelper()
    #print(skin.shape)
    prediction.loadImage1dBinary(skin, N, M)
    predictionPrior = imageHelper()
    predictionPrior.loadImage1dBinary(skin_prior, N, M)

    plt.figure(fig, figsize=(10, 7))
    plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    plt.imshow(img.image)
    plt.axis('off')
    plt.title('Test image')

    plt.subplot2grid((4, 5), (0, 2), rowspan=2, colspan=2)
    plt.imshow(prediction.image, cmap='gray')
    plt.axis('off')
    plt.title('Skin prediction')

    plt.subplot2grid((4, 5), (2, 2), rowspan=2, colspan=2)
    plt.imshow(predictionPrior.image, cmap='gray')
    plt.axis('off')
    plt.title('Skin prediction PRIOR')

    plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    plt.imshow(mask.image, cmap='gray')
    plt.axis('off')
    plt.title('GT mask')

    plt.subplot(4, 5, 5)
    plt.imshow(fpImage, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive')
    plt.subplot(4, 5, 10)
    plt.imshow(fnImage, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative')
    plt.subplot(4, 5, 15)
    plt.imshow(fpImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive PRIOR')
    plt.subplot(4, 5, 20)
    plt.imshow(fnImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative PRIOR')

    plt.savefig(f"../{fig}.pdf")
