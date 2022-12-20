import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlp_cmp import mlp, fc, activation as a, loss as l

TRAIN_SIZE = 1000
TEST_SIZE = 100
EPOCHS = 1000
L_RATE = 0.01

FC1_NEURONS = 16
FC2_NEURONS = 16
GDESCENT_TYPE = 'stoc' # 'stoc' | 'bat'
LOSS = l.ce_loss

if __name__ == "__main__":
    # Mnist dataset loaded with "fetch_openml("mnist_784")", and then
    # data and target arrays saved locally to avoid downloading it every run
    print("------Loading MNIST dataset----------")
    # 70000 elements -> numpy.ndarray
    data = np.load("./data.npy").astype(float)
    # 70000 elements -> numpy.ndarray
    target = np.load("./target.npy", allow_pickle=True).astype(float)
    categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # Normalize data
    data = data / max(data[0])
    # Make target one-hot encoded vectors from strings
    target_v = np.zeros((len(target), len(categories)))
    target_v[np.arange(len(target)), target.astype(int)] = 1
    print("----------------------------")
    print(
        f"Imported MNIST dataset, {len(data)} images"
        f" {len(target)} labels, {len(categories)} categories"
    )
    in_size = len(data[0])  # 784
    out_size = len(categories)  # 10

    perceptron = mlp.MLP(in_size, out_size, LOSS, GDESCENT_TYPE)

    fc1 = fc.FC(num_in_ft=in_size, num_out_ft=FC1_NEURONS,
                activation=a.sigmoid)
    perceptron.add_layer(fc1)

    fc2 = fc.FC(num_in_ft=FC1_NEURONS, num_out_ft=FC2_NEURONS,
                activation=a.sigmoid)
    perceptron.add_layer(fc2)

    out_l = fc.FC(num_in_ft=FC2_NEURONS, num_out_ft=out_size,
                  activation=a.softmax)
    perceptron.add_layer(out_l)
    print("------------Output Layer^--------------")

    d_train = data[:TRAIN_SIZE]
    d_train = d_train.reshape(d_train.shape[0], 1, d_train.shape[1])
    t_train = target_v[:TRAIN_SIZE]

    d_test = data[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
    t_test = target_v[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]

    perceptron.train(d_train, t_train, EPOCHS, L_RATE)
    perceptron.test(d_test, t_test)
