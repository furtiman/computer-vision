import numpy as np
from collections.abc import Callable

# Resources used:
# Confirmation of initial design and understanding of how to implement backward propagation:
# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
# In here, in addition to dL/dW discussed in lecture slides, I understood what is actually
# propagated to the next layer, that is the derivative of th error multiplied by the weights
#
#


class FC:
    """
    Fully connected layer implementation - Ivan Turasov
    """

    total_fc_num = 0

    def __init__(
        self,
        num_in_ft: int,
        num_out_ft: int,
        activation: Callable[[bool, np.array], np.array],
    ):
        """
        wm - weight matrix of size (num_in_ft x num_out_ft)
        bs - vector of all neurons' biases (size = num_out_ft)
        """
        total_fc_num += 1  # Keep track of total fc layers number

        self.num_in_ft = num_in_ft
        self.num_out_ft = num_out_ft
        self.activation = activation  # Activation function
        self.wm = np.random.default_rng(1).random((num_in_ft, num_out_ft))
        self.bs = np.zeros(1, num_out_ft)
        self.input = None
        self.out_before_activation = None
        self.output = None

    def fw(self, in_ft: np.array) -> np.array:
        """Apply weights and activation function"""
        self.input = in_ft
        self.out_before_activation = np.dot(in_ft, self.wm) + self.bs # Yin
        self.output = self.activation(prime=False, in_ft=self.out_before_activation)
        return self.output

    def bw(self, in_err: np.array, learning_rate: float) -> np.array:
        """
        Calculate error based on error of the previous layer
        activation_prime(self.out_before_activation)
        """

        # Activation derivative dY/dYin(Yin)
        activation_err = self.activation(prime=True, in_ft=self.out_before_activation)

        # Loss derivative over weights (dL/dW)
        weight_err = np.dot(self.input.T, in_err * activation_err)
        # Loss Derivative over previous layer input (dL/din_ft)
        out_err = np.dot(in_err * activation_err, self.wm.T)
        # Update weights (w1 = w0 - n * dL/dW) - after the old weight matrix was used
        self.wm = self.wm - learning_rate * weight_err

        # Mod 4 slide 80
        return out_err  # Propagated error to next layer
