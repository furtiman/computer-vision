# Multilayer Perceptron Implementation

Implementation of assignment 1 of course 5LSH0.
MNIST dataset has been saved locally 

### Usage
#### 0. Unpack data archive

Unpack the `data.zip` archive to this folder, should result in 2 files: `data.npy` and `target.npy`

#### 1. Create a virtual environment and install requirements

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### 2. Run the main loop.

`main.py` contains the main program sequence.
It also contains all the hyperparameters of the model.
There, the model is built with 2 hidden layers, but it can be extended ad-hoc.

```python
TRAIN_SIZE = 10000
TEST_SIZE = 1000
EPOCHS = 400
L_RATE = 0.01

FC1_NEURONS = 64
FC2_NEURONS = 64
BATCH_SIZE = 1 # 1 for stochastic descent
LOSS = l.ce_loss # l.ce_loss | l.mse
```

After the training, the testing wil be performed and the results will be printed in the console as follows:

```bash
BS = 1: Epoch #390 - avg loss = 0.10581184752953403, avg Accuracy = 0.9686035805626597, C: 9892, T: 10000
100%|██████████████████████████████████████████████████████████████████| 400/400 [08:24<00:00,  1.26s/it]
Total mean accuracy during training 0.9690669999999995
Accuracy = 946 / 1000 = 0.946
Avg test err = 0.24529931852941633
```