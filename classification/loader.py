import numpy as np


def load_data(file_path):
    """
    Loads data in a proper format
    """
    data_xy = np.genfromtxt(file_path, delimiter=' ', skip_header=1, dtype=str)
    x = data_xy[:, 2:]
    x = x.astype(np.float32)

    y = np.squeeze(data_xy[:, 1])
    y[np.where(y == '"M"')] = 1
    y[np.where(y == '"B"')] = 0
    y = y.astype(np.int8)
    return x, y


def standardize(train_set, valid_set):
    """
    Standardize train set and uses its mean and std
    for validation data
    """
    train_x = train_set[0]
    valid_x = valid_set[0]

    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)

    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std

    return {'train': (train_x, train_set[1]), 'valid': (valid_x, valid_set[1])}