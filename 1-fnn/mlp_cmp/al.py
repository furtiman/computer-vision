import numpy as np
from collections.abc import Callable
from . import activation as a

# Resources used:
# Confirmation of initial design and understanding of how to implement backward propagation:
# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
# In here, in addition to dL/dW discussed in lecture slides, I understood what is actually
# propagated to the next layer, that is the derivative of th error multiplied by the weights
#


class AL:
    """
    Activation layer implementation - Ivan Turasov
    """

    total_al_num = 0

    def __init__(
        self,
        activation: Callable[[bool, np.array], np.array],
    ):
        """
        wm - weight matrix of size (num_in_ft x num_out_ft)
        bs - vector of all neurons' biases (size = num_out_ft)
        """
        AL.total_al_num += 1  # Keep track of total fc layers number

        self.activation = activation
        # self.activation_prime = activation_prime
        print(
            f"---------Init Activation Layer {self.activation}-----------"
        )

    def fw(self, fw_in: np.array) -> np.array:
        """Apply activation function"""
        self.input = fw_in
        self.fw_out = self.activation(prime=False, in_ft=self.input)
        return self.fw_out

    def bw(self, bw_in: np.array, learning_rate: float) -> np.array:
        bw_out = self.activation(prime=True, in_ft=self.input) * bw_in
        return bw_out
