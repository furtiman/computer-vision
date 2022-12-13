import numpy as np
from collections.abc import Callable

def gdescent(gradient: Callable[[np.array], np.array], start, learn_rate, n_iter):
    vector = start
    
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
    
    return vector
    