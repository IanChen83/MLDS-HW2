import config
import param
import cost_function
import theano
__author__ = 'patrickchen'

grad = []


def initialize_grad():
    for i in config.batch_num:
        grad.append(theano.grad(cost=cost_function.cost[i],
                                wrt=param.W + param.B
                                )
                    )
