from nnet.hidden_layer import HiddenLayer
from nnet.softmax_layer import SoftmaxLayer
import theano
import theano.tensor as T
import numpy as np
from theano import Param
from nnet.train_set_iterator import TrainSetIterator


class MLP(object):
    """
    MLP class, which contains layers and functionality
    for training and testing
    """

    def __init__(self, n_in, n_hidden, n_out, dropout_prob, learning_rate, weight_decay):

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.training_mode = T.iscalar('training_mode')

        rng = np.random.RandomState(1234)

        self.hiddenLayer = HiddenLayer(rng=rng, input=self.x,
                                       n_in=n_in, n_out=n_hidden)

        self.outputLayer = SoftmaxLayer(rng=rng,
                                        input=self.hiddenLayer.output,
                                        n_in=n_hidden, n_out=n_out,
                                        training_mode=self.training_mode,
                                        dropout_prob=dropout_prob)

        self.params = self.hiddenLayer.params + self.outputLayer.params

        L2_reg = (self.hiddenLayer.W ** 2).sum() + (self.outputLayer.W ** 2).sum()
        cost = self.outputLayer.cost(self.y) + weight_decay * L2_reg
        grads = T.grad(cost, self.params)
        updates = self._vanilla_updates(grads, learning_rate)

        self.train_model = theano.function([self.x, self.y, Param(self.training_mode, default=1)], cost,
                                           updates=updates)

        tp, tn, fp, fn = self.outputLayer.confusion_matrix(self.y)
        self.validate_model = theano.function([self.x, self.y, Param(self.training_mode, default=0)],
                                              [self.outputLayer.p_y_given_x[:, 1], tp, tn, fp, fn])


    def test_mlp(self, datasets, n_epochs, batch_size, roc_file):
        """
        Performs training of MLP and its testing.
        Writes probabilities of test cases into file,
        which is used for ROC analysis
        """

        train_iterator = TrainSetIterator(datasets['train'], batch_size)
        valid_set_x, valid_set_y = datasets['valid']

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for train_x, train_y in train_iterator:
                self.train_model(train_x, train_y)

        res = self.validate_model(valid_set_x, valid_set_y)
        pairs_y = zip(res[0], valid_set_y)
        roc_file.write('\n'.join('%s %s' % x for x in pairs_y))
        roc_file.write('\n')
        print res[1:5]
        return np.array(res[1:5])

    def _vanilla_updates(self, grads, learning_rate):
        updates = []
        for param_i, grad_i in zip(self.params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))
        return updates