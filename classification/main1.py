import numpy as np
from loader import load_data, standardize
from classification.mlp1 import MLP

n_in = 10
n_hidden = 25
n_out = 2
dropout_prob = 0.5
learning_rate = 0.1
n_epochs = 500
batch_size = 10
weight_decay = 0.0001
K = 10

x, y = load_data('../data1.txt')
roc_file = open('roc.txt', 'w')

step = len(x) / K + 1
confusion_matrix = np.array([0, 0, 0, 0])
for i in np.arange(0, len(x) + 1, step):
    valid_set = x[i:i + step, :], y[i:i + step]
    train_set = np.delete(x, range(i, i + step), axis=0), np.delete(y, range(i, i + step))
    datasets = standardize(train_set, valid_set)

    ml = MLP(n_in, n_hidden, n_out, dropout_prob, learning_rate, weight_decay)
    confusion_matrix += ml.test_mlp(datasets, n_epochs, batch_size, roc_file)

print 'tp tn fp fn : ', confusion_matrix
roc_file.close()