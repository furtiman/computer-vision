from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


if __name__ == '__main__':
    mnist = load_digits()
    print(type(mnist))
