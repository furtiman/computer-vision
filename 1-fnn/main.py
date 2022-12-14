from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from mlp_cmp import mlp, fc, activation as a, loss as l

FC1_NEURONS = 16
FC2_NEURONS = 16

if __name__ == "__main__":
    print("------Loading MNIST dataset----------")
    mnist = fetch_openml("mnist_784")
    # 70000 elements (pandas.core.frame.DataFrame -> numpy.ndarray)
    data = mnist.data.to_numpy()

    # 70000 elements ( pandas.core.series.Series -> numpy.ndarray
    target = mnist.target.to_numpy()
    categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    print("----------------------------")
    print(
        f"Imported MNIST dataset, {len(data)} images"
        f" {len(target)} labels, {len(categories)} categories"
    )
    in_size = len(data[0])  # 784
    out_size = len(categories)  # 10

    perceptron = mlp.MLP(in_size, out_size)
    print("------MLP object initialised---------")
    fc1 = fc.FC(num_in_ft=in_size, num_out_ft=FC1_NEURONS, activation=a.sigmoid)
    fc2 = fc.FC(num_in_ft=in_size, num_out_ft=FC2_NEURONS, activation=a.softmax)
    perceptron.add_layer(fc1)
    print("-------------Added FC1---------------")
    perceptron.add_layer(fc2)
    print("-------------Added FC2---------------")
