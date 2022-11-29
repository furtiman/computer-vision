import numpy as np

class FC:
    '''
        Fully connected layer implementation - Ivan Turasov
    '''

    def __init__(self, in_features: int, out_features: int) -> FC:
        '''
            wm - weight matrix of size (in_features x out_features)
            bs - vector of all neurons' biases
        '''
        self.wm = np.random.default_rng(1).random((in_features, out_features))
        self.bs = np.zeros(out_features)
