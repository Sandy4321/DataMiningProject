import numpy as np

def load_data(file_path):
    data_xy = np.genfromtxt(file_path, delimiter=' ', skip_header=1, dtype=str)
    x = data_xy[:, 1:23]
    x = x.astype(np.float32)
    y = np.squeeze(data_xy[:, 23])
    y = y.astype(np.float32)
    return x, y
