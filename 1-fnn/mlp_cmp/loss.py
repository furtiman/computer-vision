import numpy as np
import torch

# https://lucasdavid.github.io/blog/machine-learning/crossentropy-and-logits/#losses


def ce_loss(prime: bool, p: np.array, t: np.array):
    """
    Cross-Entropy loss implementation
    Resources used for understanding and inspiration:
        https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient

    In mod4 slide 48 the result of the loss (1.599) can be achieved with log base 2
    However most of library implementation (torch, sklearn) use natural logarithm
    In this implementation natural log is usen (np.log())
    """

    e = np.finfo(float).eps  # Make sure we dont take log of 0

    if not prime:
        loss = -np.sum(t * np.log(p + e))
    else:
        loss = p - t
    return loss


def mse(prime: bool, p: np.array, t: np.array):
    """
    Mean Squared Error loss function
    """
    if not prime:
        return np.mean(np.power(t - p, 2))
    else:
        return 2 * (p - t) / t.size
