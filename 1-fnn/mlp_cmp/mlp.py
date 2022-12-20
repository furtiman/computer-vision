import numpy as np
from tqdm import tqdm # Fro status bar during training
from . import activation, loss, fc
from collections.abc import Callable
import matplotlib.pyplot as plt


class MLP:
    def __init__(
        self,
        in_size: int,
        out_size: int,
        loss: Callable[
            [(bool, np.array, np.array)],
        ],
        gd_type: str = "stoc",
        bat_size: int = 1,
    ) -> None:
        """
        gd_type: 'stoc' - Stochastic|'mbat' - Minibatch|'bat' - Batch
        """
        self.in_size = in_size
        self.out_size = out_size
        self.gd_type = gd_type
        self.bat_size = bat_size
        self.loss = loss

        self.layers = []
        print("------MLP object initialized---------")

    def add_layer(self, layer: fc.FC) -> None:
        self.layers.append(layer)

    def predict(self, sample: np.array) -> np.array:
        """Return array of confidence scores for one provided sample"""
        out = sample
        for l in self.layers:
            out = l.fw(out)
        return out

    def train(
        self,
        train_samples: np.array,
        train_labels: np.array,
        epochs: int,
        l_rate: float,
    ):

        num_s = len(train_samples)
        batch_loss = 0
        loss_curve = []

        for e in tqdm(range(epochs)):
            err = 0
            if self.gd_type == "stoc":
                for i in range(num_s):
                    p = self.predict(train_samples[i])
                    t = train_labels[i]

                    err += self.loss(prime=False, p=p, t=t)
                    gradient = self.loss(prime=True, p=p, t=t)
                    for l in reversed(self.layers):
                        # Weights are updated in every layer, so the end gradient is not needed
                        gradient = l.bw(gradient, l_rate)
                        # print(gradient)
                err /= num_s
                loss_curve.append(err)
            elif self.gd_type == "bat":
                gradient = 0

                # Forward - accumulate loss and gradient of all training set
                for i in range(num_s):
                    # Go through all FC layers
                    p = self.predict(train_samples[i])
                    t = train_labels[i]

                    err += self.loss(prime=False, p=p, t=t)
                    gradient += self.loss(prime=True, p=p, t=t)
                err /= num_s
                gradient /= num_s
                # Backward
                for l in reversed(self.layers):
                    # Weights are updated in every layer, so the
                    # end gradient is not needed
                    gradient = l.bw(gradient, l_rate)

            if e%100 == 0:
                    print(f"{self.gd_type}: Epoch #{e} - avg CE = {err}")
        plt.plot(loss_curve)
        plt.show()


    def validate():
        pass

    def test(
        self,
        test_samples: np.array,
        test_labels: np.array,
    ):
        err = 0
        for i, s in enumerate(test_samples):
            p = self.predict(s)
            t = test_labels[i]
            p_class = np.argmax(p) # index of prediction
            t_class = np.argmax(t) # index of label
            err += self.loss(prime=False, p=p, t=t)
        print(f"Avg test err = {err / len(test_samples)}")
