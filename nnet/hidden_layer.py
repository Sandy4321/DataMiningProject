import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out):
        self.input = input
        W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype='float32')

        W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((n_out,), dtype='float32')
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = T.tanh(lin_output)
        self.params = [self.W, self.b]
