import numpy as np
from loader import load_data
from regression.mlp2 import MLP

n_in = 22
n_hidden = 40
n_out = 1
dropout_prob = 0.5
learning_rate = 0.1
n_epochs = 500
batch_size = 10
weight_decay = 0.00001
K = 10

x, y = load_data('../data2.txt')

step = len(x) / K + 1
mse = 0.0
for i in np.arange(0, len(x) + 1, step):
    valid_set = x[i:i + step, :], y[i:i + step]
    train_set = np.delete(x, range(i, i + step), axis=0), np.delete(y, range(i, i + step))
    datasets = {'train': train_set, 'valid': valid_set}

    ml = MLP(n_in, n_hidden, n_out, dropout_prob, learning_rate, weight_decay)
    mse += ml.test_mlp(datasets,n_epochs, batch_size)

print 'mse:', mse/K