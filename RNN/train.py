import theano

import param
import config
import update

__author__ = 'patrickchen'

train = None


def initialize_train():
    if len(param.grad) == 0:
        param.initialize_grad()

    if update.update is None:
        update.initialize_update()

    train = theano.function(inputs=[param.X, param.Y],
                            outputs=param.cost,
                            updates=update.update
                            )
