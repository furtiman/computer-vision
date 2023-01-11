import numpy as np
import torch

# https://lucasdavid.github.io/blog/machine-learning/crossentropy-and-logits/#losses

def ce_loss(prime: bool, p: np.array, t: np.array):
    '''
        Cross-Entropy loss implementation
        Resources used for understanding and inspiration:
            https://vitalflux.com/cross-entropy-loss-explained-with-python-examples/
        
        In mod4 slide 48 the result of the loss (1.599) can be achieved with log base 2
        However most of library implementation (torch, sklearn) use natural logarithm
        In this implementation natural log is usen (np.log())
    '''
    # Since both target and prediction are np arrays, we can compute
    # Hadamard product of them and get a sum of elements after
    # p = p.clip(min=1e-8, max=None)
    if not prime:
        loss = -np.sum(t * np.log(p))
    else:
        loss = t * (-1 / p)
    return loss

# loss function and its derivative
def mse(prime: bool, p: np.array, t: np.array):
    if not prime:
        return np.mean(np.power(t - p, 2))
        # return np.mean(np.power(t - p, 2))
    else:
        return 2*(p - t)/t.size
