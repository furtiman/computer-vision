import numpy as np
from tqdm import tqdm  # Fro status bar during training
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
        # gd_type: str = "stoc",
        bat_size: int = 1,
    ) -> None:
        """
        gd_type: 'stoc' - Stochastic|'mbat' - Minibatch|'bat' - Batch
        """
        self.in_size = in_size
        self.out_size = out_size
        # self.gd_type = gd_type
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
        accuracy_sum = 0

        indices = np.random.permutation(
            train_samples.shape[0]
        )  # Shuffle indices of data

        for e in tqdm(range(epochs)):
            total_p = 0
            correct_p = 0
            # if self.bat_size != 10000:
            # Prepare batches
            for i in range(0, len(indices), self.bat_size):  # Equal to train size
                ids = indices[i : i + self.bat_size]
                batch_samples = train_samples[ids]
                batch_labels = train_labels[ids]
                err = 0
                gradient = 0

                for i in range(len(batch_samples)):
                    total_p += 1
                    p = batch_samples[i].reshape(len(batch_samples[i]), 1).T
                    p = self.predict(p)

                    t = batch_labels[i]

                    if np.argmax(p) == np.argmax(t):
                        correct_p += 1

                    err += self.loss(prime=False, p=p, t=t)
                    gradient += self.loss(prime=True, p=p, t=t)

                # # Stops working when the average of gradient is taken
                # gradient /= len(batch_samples)
                err /= len(batch_samples)

                for l in reversed(self.layers):
                    # Weights are updated in every layer, so the end gradient is not needed
                    gradient = l.bw(gradient, l_rate)

            accuracy_sum += correct_p / total_p
            if e % 10 == 0:
                print(
                    f"BS = {self.bat_size}: Epoch #{e} - avg loss = {err}"
                    f", avg Accuracy = {accuracy_sum / (e + 1)}"
                    f", C: {correct_p}, T: {total_p}"
                )
        print(f"Total mean accuracy during training {accuracy_sum / epochs}")

    def validate():
        pass

    def test(
        self,
        test_samples: np.array,
        test_labels: np.array,
    ):
        err = 0
        total = 0
        correct = 0
        for i, s in enumerate(test_samples):
            total += 1
            p = self.predict(s)
            t = test_labels[i]
            # p_class = np.argmax(p) # index of prediction
            # t_class = np.argmax(t) # index of label
            if np.argmax(p) == np.argmax(t):
                correct += 1
            err += self.loss(prime=False, p=p, t=t)
        print(f"Accuracy = {correct} / {total} = {correct / total}")
        print(f"Avg test err = {err / len(test_samples)}")
