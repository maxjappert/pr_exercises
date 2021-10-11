import numpy as np
from imageHelper import imageHelper


def get_prior(mask: imageHelper) -> (float, float):
    [N, M] = mask.shape
    image_mask = mask.image[:]
    # TODO: EXERCISE 2 - Compute the skin and nonskin prior
    prior_skin = None
    prior_nonskin = None
    return prior_skin, prior_nonskin
