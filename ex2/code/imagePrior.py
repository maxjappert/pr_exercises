import numpy as np
from imageHelper import imageHelper


def get_prior(mask: imageHelper) -> (float, float):
    [N, M] = mask.shape
    image_mask = mask.getLinearImageBinary().astype(int)[:, 0]
    # EXERCISE 2 - Compute the skin and nonskin prior

    # This is done as explained in the theory answers.
    prior_skin = float(np.sum(image_mask)) / (N*M)
    prior_nonskin = 1.0 - prior_skin

    return prior_skin, prior_nonskin
