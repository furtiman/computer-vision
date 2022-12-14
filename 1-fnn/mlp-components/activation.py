import numpy as np
import torch

def relu(prime: bool, in_ft: np.array, fw_out: np.array=None) -> np.array:
    ''' ReLU:
        f(x) = max(0, x)
        f'(x) = (0 if x <= 0) , (1 if x > 0)
    '''
    if not prime:
        return np.maximum(0, in_ft)
    else:
        rc = 0 if in_ft <= 0 else 1
        return rc

def softplus(prime: bool, in_ft: np.array, fw_out: np.array=None) -> np.array:
    ''' SoftPlus:
        f(x) = ln(1 + e^x)
        f'(x) = e^x / 1 + e^x
    '''
    e_x = np.exp(in_ft)
    if not prime:
        return np.log(1 + e_x)
    else:
        return e_x / (1 + e_x)

def sigmoid(prime: bool, in_ft: np.array, fw_out: np.array=None) -> np.array:
    ''' Sigmoid:
        f(x) = 1 / 1 + e^(-x)
        f'(x) = (1 / 1 + e^(-x)) * (1 - (1 / 1 + e^(-x)))
    '''
    sig = 1 / (1 + np.exp( - in_ft))
    if not prime:
        return sig
    else:
        return sig * (1 - sig)

def softmax(prime: bool, in_ft: np.array, fw_out: np.array=None) -> np.array:
    '''
        Softmax:
        f(x) = e^x / SUMy->N(e^y), where N is the number of vector elements
        Details with "axis = 0" and using the np.max in nominator
        used based on discussion here -
        https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

        Output of forward softmax is needed to compute the input gradient
        f'(x) inspired by and understood with help of:
            https://e2eml.school/softmax.html
            https://gitlab.com/brohrer/cottonwood/
            https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    '''
    if not prime:
        e_x = np.exp(in_ft - np.max(in_ft))
        return e_x / e_x.sum(axis = 0, keepdims = True)
    else:
        # Create two 2D arrays with 1 row to create a Jacobian matrix
        fw_out = np.reshape(fw_out, (1, -1))
        grad = np.reshape(in_ft, (1, -1))

        d_softmax = (
            fw_out * np.identity(fw_out.size)
            - np.dot(fw_out.T, fw_out))
        
        return np.dot(grad, d_softmax).ravel()

def test_softmax(in_ft: np.array, test_num: int):
    # TODO: test softmax derivative too
    # 6 decimals provide same results, after 7th decimal assertion does not pass
    ROUND_DECIMALS = 6

    # Calculate ours
    res = softmax(prime=False, in_ft=in_ft)
    res = np.round(res, ROUND_DECIMALS) # Our result

    # Calculate theirs
    in_ft_t = torch.tensor(in_ft).float() # Same list converted to tensor for torch
    check = torch.nn.functional.softmax(in_ft_t, dim=0).numpy()
    check = np.round(check.tolist(), ROUND_DECIMALS) # PyTorch result

    assert np.array_equal(res, check), f"Softmax {test_num} failed \n Ours:   {res} \
                                        \n Theirs: {check}"


if __name__ == "__main__":
    # Check softmax forward propagation with results from PyTorch as baseline
    ### Case 1
    in_ft = [1, 3, 2]
    test_softmax(in_ft, 1)
    
    ### Case 2
    in_ft = [5, 2, 8, 9]
    test_softmax(in_ft, 2)
