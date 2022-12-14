import numpy as np
from collections.abc import Callable

# Resources used:
# Confirmation of initial design and understanding of how to implement backward propagation:
# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
# In here, in addition to dL/dW discussed in lecture slides, I understood what is actually
# propagated to the next layer, that is the derivative of th error multiplied by the weights
#


class FC:
    """
    Fully connected layer implementation - Ivan Turasov

    In this implementation the activation function is a part of the FC layer,
    rather than a separate layer. Hence it is called as part of fw and bw
    pass of the FC layer.
    This choice was made due to the explanation in lectures, and even though
    it brings another layer of complexity when other types of layers would be
    implemented, for this particular assignment this seemed the best ways to
    fully understand the full backward propagation.
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
        FC.total_fc_num += 1  # Keep track of total fc layers number

        self.num_in_ft = num_in_ft
        self.num_out_ft = num_out_ft
        self.activation = activation  # Activation function
        self.wm = np.random.default_rng(1).random((num_in_ft, num_out_ft))
        self.bs = np.zeros((1, num_out_ft))
        self.input = None
        self.logits = None
        self.output = None

    def fw(self, fw_in: np.array) -> np.array:
        """Apply weights and activation function"""
        self.input = fw_in
        self.logits = np.dot(fw_in, self.wm) + self.bs # Yin
        self.fw_out = self.activation(prime=False,
                                      in_ft=self.logits)
        return self.fw_out

    def bw(self, bw_in: np.array, learning_rate: float) -> np.array:
        """
        Calculate error based on error of the previous layer
        activation_prime(self.logits)
        """

        # Activation derivative dY/dYin(Yin)
        activation_err = self.activation(prime=True,
                                         in_ft=self.logits, fw_out=self.fw_out)

        # Loss derivative over weights (dL/dW)
        weight_err = np.dot(self.input.T, bw_in * activation_err)

        # Loss Derivative over previous layer input (dL/din_ft)
        # Multiplying by wight matrix, because the derivative is wrt in_ft
        # and by chain rule, the last term is d(out_before_act)/din_ft = weights
        bw_out = np.dot(bw_in * activation_err, self.wm.T)

        # Update weights (w1 = w0 - n * dL/dW) - after the old weight matrix was used
        self.wm = self.wm - learning_rate * weight_err

        # Mod 4 slide 80
        return bw_out  # Propagated error to next layer
