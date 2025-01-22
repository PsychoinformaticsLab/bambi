import numpy as np
from scipy.stats import truncnorm

def truncatedprior(mean, std, lower, upper, size=1):
    """
    Generate truncated normal distribution samples.

    Parameters:
    - mean (float): Mean of the normal distribution.
    - std (float): Standard deviation of the normal distribution.
    - lower (float): Lower bound of the truncated distribution.
    - upper (float): Upper bound of the truncated distribution.
    - size (int): Number of samples to generate.

    Returns:
    - ndarray: Samples from the truncated normal distribution.
    """
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm(a, b, loc=mean, scale=std).rvs(size)
