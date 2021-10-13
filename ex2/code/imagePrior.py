import numpy as np
from imageHelper import imageHelper


def get_prior(mask: imageHelper) -> (float, float):
    [N, M] = mask.shape
    image_mask = mask.image[:]
    # TODO: EXERCISE 2 - Compute the skin and nonskin prior

    skinpixel_counter = 0
    nonskinpixel_counter = 0

    for i in range(0, N):
        for j in range(0, M):
            if image_mask[i, j] > 0.5:
                skinpixel_counter += 1
            else:
                nonskinpixel_counter += 1

    prior_skin = skinpixel_counter / (N*M)
    prior_nonskin = nonskinpixel_counter / (N*M)
    return prior_skin, prior_nonskin
