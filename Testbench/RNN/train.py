import theano

import param
import update

__author__ = 'patrickchen'

train = None
cost = None
dimshuffle = None


def initialize_train():
    global train, cost
    if len(param.grad) == 0:
        param.initialize_grad()

    if update.update is None:
        update.initialize_update()

    train = theano.function(inputs=[param.X, param.Y],
                            outputs=param.cost,
                            updates=update.update,
                            allow_input_downcast=True,
                            )

    cost = theano.function(inputs=[param.X, param.Y],
                           outputs=param.cost,
                           allow_input_downcast=True,
                           )
    dimshuffle = theano.function(inputs=[param.X],
                                 outputs=param.X.dimshuffle(1,0,2),
                                 allow_input_downcast=True
                                 )