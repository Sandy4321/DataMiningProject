import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class LinearRegressionLayer(object):
    def __init__(self, rng, input, n_in, training_mode, dropout_prob):
        self.W = theano.shared(value=np.zeros((n_in, 1), dtype='float32'),
                               name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((1,), dtype='float32'),
                               name='b', borrow=True)
        inv_dropout_prob = np.float32(1.0 - dropout_prob)
        self.y_pred = ifelse(T.eq(training_mode, 1),
                             T.dot(self._dropout(rng, input, dropout_prob), self.W) + self.b,
                             T.dot(input, inv_dropout_prob * self.W) + self.b)
        self.params = [self.W, self.b]

    def _dropout(self, rng, layer, p):
        """
        Generates dropout masks
        """
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
        output = layer * T.cast(mask, 'float32')
        return output

    def cost(self, y):
        """
        MSE
        """
        return T.mean((y-T.transpose(self.y_pred))**2)
