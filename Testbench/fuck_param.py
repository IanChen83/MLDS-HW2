import theano.tensor as T
import theano
import numpy
import numpy.random
import fuck_rnn_config as config
__author__ = 'patrickchen'

X = T.ftensor3()
X_shuffle = X.swapaxes(0, 1)
Y = T.ftensor3()
Y_shuffle = Y.swapaxes(1, 0)

Wi = theano.shared(numpy.random.normal(
    loc=config.random_mu,
    scale=config.random_sigma,
    size=(config.input_dim, config.hidden_layer_dim)
))
Wh = theano.shared(
    numpy.identity(config.hidden_layer_dim)
)
Wo = theano.shared(numpy.random.normal(
    loc=config.random_mu,
    scale=config.random_sigma,
    size=(config.hidden_layer_dim, config.output_dim)
))
Bh = theano.shared(numpy.random.normal(
    loc=config.random_mu,
    scale=config.random_sigma,
    size=(config.batch_num, config.hidden_layer_dim)
))
Bo = theano.shared(numpy.random.normal(
    loc=config.random_mu,
    scale=config.random_sigma,
    size=(config.batch_num, config.output_dim)
))

parameters = [Wi, Wh, Wo, Bh, Bo]
