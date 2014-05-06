import numpy as np


class TrainSetIterator(object):
    """
    Yields batches of random
    samples from a training set
    """

    def __init__(self, dataset, batch_size):
        x, y = dataset
        self.rng = np.random.RandomState(42)
        idx = np.arange(x.shape[0])
        self.rng.shuffle(idx)
        self.x = x[idx, :]
        self.y = y[idx]
        self.index = 0
        self.batch_size = batch_size
        self.size = self.x.shape[0]


    def __iter__(self):
        return self

    def next(self):
        begin = self.index
        end = self.index + self.batch_size
        if end <= self.size:
            x = self.x[begin:end, :]
            y = self.y[begin:end]
            self.index += self.batch_size
            return x, y
        else:
             self.__restart()
             raise StopIteration


    def __restart(self):
        idx = np.arange(self.size)
        self.rng.shuffle(idx)
        self.x = self.x[idx, :]
        self.y = self.y[idx]
        self.index = 0
