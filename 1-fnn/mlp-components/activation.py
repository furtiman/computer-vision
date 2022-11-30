import numpy as np

def relu(in_ft: np.array) -> np.array:
    ''' ReLU: f(x) = max(0, x) '''
    return np.maximum(0, in_ft)

def softplus(in_ft: np.array) -> np.array:
    ''' SoftPlus: f(x) = ln(1 + e^x) '''
    return np.log(1 + np.exp(in_ft))

def sigmoid(in_ft: np.array) -> np.array:
    ''' Sigmoid: f(x) = 1 / 1 + e^(-x) '''
    return 1 / (1 + np.exp(-in_ft))

def softmax(in_ft: np.array) -> np.array:
    '''
        Softmax: f(x) = e^x / SUMy->N(e^y), where N is the number of vector elements
        Details with "axis = 0" and using the np.max in nominator
        used based on discussion here -
        https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    '''
    e_x = np.exp(in_ft - np.max(in_ft))
    return e_x / e_x.sum(axis = 0, keepdims = True)
