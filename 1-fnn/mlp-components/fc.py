import numpy as np
from collections.abc import Callable

class FC:
    '''
        Fully connected layer implementation - Ivan Turasov
    '''
    total_fc_num = 0

    def __init__(self, num_in_ft: int, num_out_ft: int, activation: Callable[[np.array], np.array]):
        '''
            wm - weight matrix of size (num_in_ft x num_out_ft)
            bs - vector of all neurons' biases (size = num_out_ft)
        '''
        total_fc_num += 1 # Keep track of total fc layers number

        self.num_in_ft = num_in_ft
        self.num_out_ft = num_out_ft
        self.activation = activation # Activation function
        self.wm = np.random.default_rng(1).random((num_in_ft, num_out_ft))
        self.bs = np.zeros(1, num_out_ft)

    def fw(self, in_ft: np.array) -> np.array:
        ''' Apply weights and activation function '''
        out = np.dot(in_ft, self.wm) + self.bs
        return self.activation(out)
    
    def bw(self):
        pass