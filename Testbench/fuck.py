import fuck_rnn_config as config
import fuck_param as param
import data
import theano
import theano.tensor as tensor
import numpy
try:
    from itertools import izip
except ImportError:
    izip = zip
__author__ = 'patrickchen'

rnn_test_cost = None
rnn_test_y_evaluate = None
rnn_train = None
rnn_train_grad = None


def __do_nothing__(_):
    pass


def fuck(par):
    initialize_train()
    cycle = 0
    try:
        cycle = int(par[0])
        output_name = par[1]
    except ValueError:
        print("cycle is not a number")
    if cycle == 0:
        cycle = 1
    train(cycle)


def cost_func(y_e, y):
    return (y_e - y).norm(2)
    # return (y_e - y).norm(2, axis=1)
    # return (y_e - y).norm(1, axis=1)


def act_func(func):
        return (1 + tensor.tanh(func / 2)) / 2

"""
    #################### Model ######################################
"""


def initialize_train():
    global rnn_train, rnn_test_cost, rnn_test_y_evaluate, rnn_train_grad

    def step(x_t, a_tm1, y_tml):
            a_t = act_func(tensor.dot(x_t, param.Wi) + tensor.dot(a_tm1, param.Wh) + param.Bh)
            y_t = tensor.dot(a_t, param.Wo) + param.Bo
            return a_t, y_t

    a_0 = theano.shared(numpy.zeros((config.batch_num, config.hidden_layer_dim)))
    y_0 = theano.shared(numpy.zeros((config.batch_num, config.output_dim)))

    [a_seq, y_seq], _ = theano.scan(
                            step,
                            sequences=param.X_shuffle,
                            outputs_info=[a_0, y_0],
                            truncate_gradient=-1
                    )
    cost = cost_func(y_seq, param.Y_shuffle)
    grad = theano.grad(cost, param.parameters)

    rnn_test_cost = theano.function(
        inputs=[param.X, param.Y],
        outputs=cost,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    rnn_test_y_evaluate = theano.function(
        inputs=[param.X],
        outputs=y_seq,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    rnn_train = theano.function(
        inputs=[param.X, param.Y],
        outputs=cost,
        updates=RMSprop(param.parameters, grad),
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    rnn_train_grad = theano.function(
        inputs=[param.X, param.Y],
        outputs=grad,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )


def update_pairs(parameter, gradient):
    parameters_updates = [(p, p - config.learning_rate * tensor.clip(g, -10, 10)) for p, g in izip(parameter, gradient)]
    return parameters_updates


def RMSprop(params, grad):
    updates = []
    for p, g in zip(params, grad):
        acc = theano.shared(p.get_value() * 0.)
        # print acc.get_value()
        acc_new = config.rho * acc + (1 - config.rho) * g ** 2
        gradient_scaling = theano.tensor.sqrt(acc_new + config.epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - config.learning_rate * g))
    return updates

"""
    #################### Train and Test #############################
"""


def train(cycle):
    for i in range(cycle):
        if rnn_train is None:
            initialize_train()
        c = data.training_input_random_selection(config.batch_num, 0, config.training_segment)
        cost = rnn_train(c[0], c[1])
        print("Cost", cost)


def test(param):
    pass
