import cPickle
import theano
from theano import tensor as T
import numpy
from util import print_error

import cost_function
import config
from activation_function import act

__author__ = 'patrickchen'

X = T.fmatrix()
Y = T.fmatrix()
Y_evaluated = None

# 'i' stands for input
# 'h' stands for history
# 'o' stands for output
Wi = []
Wh = []
Bh = []
Wo = []
Bo = []

parameters = None

step_y = None
step_a = []

cost = None
get_cost = None
grad = []


def load_param_from_file(filename):
    global parameters
    try:
        # Load config data from cPickle file
        i_parm_data = open(filename + "_I.txt", 'rb')
        conf = cPickle.load(i_parm_data)

        # Load W from cPickle file
        w_parm_data = file(filename + "_W.txt", 'rb')
        w_param = cPickle.load(w_parm_data)

        # Load B from cPickle file
        b_parm_data = file(filename + "_B.txt", 'rb')
        b_param = cPickle.load(b_parm_data)

    except IOError:
        print_error("File %s_X not found. Do nothing." % filename)
        return False

    config.input_dim = conf.input_dim
    config.output_dim = conf.output_dim
    config.hidden_layer_dim_list = conf.lidden_layer_dimension_num_list
    config.batch_num = conf.batch_num
    config.layer_num = conf.layer_num

    for i in range(config.layer_num):
        Wi[i].set_value(w_param[i].get_value())
        Wh[i].set_value(w_param[i + config.layer_num].get_value())
        Wo[i].set_value(w_param[i + config.layer_num * 2].get_value())
        Bh[i].set_value(b_param[i].get_value())
        Bo[i].set_value(b_param[i + config.layer_num].get_value())

    parameters = Wi + Wh + Wo + Bh + Bo
    return True


def write_param(filename):
    conf = config.DumpConfig()
    conf.input_dim = config.input_dim
    conf.output_dim = config.output_dim
    conf.hidden_layer_dimension_list = config.hidden_layer_dim_list
    conf.batch_num = config.batch_num
    conf.layer_num = config.layer_num

    try:
        f = file(filename + '_I.txt', 'wb')
        cPickle.dump(conf, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        f = file(filename + '_W.txt', 'wb')
        cPickle.dump(Wi + Wh + Wo, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        f = file(filename + '_B.txt', 'wb')
        cPickle.dump(Bh + Bo, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
    except IOError:
        print_error("File %s_X can't be open. Do nothing" % filename)
        return False

    return True


def initialize_param(force=False):
    # if (len(Wi) != 0 or len(Wh) != 0 or len(Wo) != 0 or len(Bh) != 0 or len(Bo) != 0)\
    #         and force is False:
    #     print_error("Some parameters have been initialized. Specify 'force' to initialize parameters explicitly.")
    #     return False
    global parameters
    if initialize_wi(force) is False\
            or initialize_wh(force) is False\
            or initialize_wo(force) is False\
            or initialize_bh(force) is False\
            or initialize_bo(force) is False:
        return False
    parameters = Wi + Wh + Wo + Bh + Bo
    return True


def initialize_step_y():
    global step_y
    step_y = theano.shared(numpy.zeros(config.output_shape()))


def initialize_step_a():
    global step_a
    temp = config.hidden_layer_dim_list + [config.output_dim]
    for i in range(len(temp)):
        step_a.append(theano.shared(value=numpy.zeros(temp[i], temp[i + 1]),
                                    name="step_a%d" % i).astype(dtype=theano.config.floatX)
                      )

"""
################### Initialize Model Parameters ############################
"""


def initialize_wi(force=False):
    if len(Wi) != 0 and force is False:
        print_error("Wi has been initialized. Specify 'force' to initialize Wi explicitly.")
        return False

    temp = [config.input_dim] + config.hidden_layer_dim_list + [config.output_dim]
    for i in range(config.layer_num):
        wp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (temp[i], temp[i + 1])).astype(dtype=theano.config.floatX)
        Wh.append(theano.shared(wp, name="Wh%d" % i))
    return True


def initialize_wh(force=False):
    if len(Wh) != 0 and force is False:
        print_error("Wh has been initialized. Specify 'force' to initialize Wh explicitly.")
        return False

    temp = config.hidden_layer_dim_list + [config.output_dim]
    for i in range(config.layer_num):
        wp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (temp[i], temp[i])).astype(dtype=theano.config.floatX)
        Wh.append(theano.shared(wp, name="Wh%d" % i))
    return True


def initialize_wo(force=False):
    if len(Wo) != 0 and force is False:
        print_error("W has been initialized. Specify 'force' to initialize W explicitly.")
        return False

    temp = [config.input_dim] + config.hidden_layer_dim_list + [config.output_dim]
    for i in range(config.layer_num):
        wp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (temp[i + 1], temp[i + 1])).astype(dtype=theano.config.floatX)
        Wo.append(theano.shared(wp, name="Wo%d" % i))
    return True


def initialize_bh(force=False):
    if len(Bh) != 0 and force is False:
        print_error("B has been initialized. Specify 'force' to initialize W explicitly.")
        return False
    temp = [config.input_dim] + config.hidden_layer_dim_list + [config.output_dim]
    for i in range(config.layer_num):
        bp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (1, temp[i + 1])).astype(dtype=theano.config.floatX)
        Bh.append(theano.shared(bp, name="Bh%d" % i))
    return True


def initialize_bo(force=False):
    if len(Bo) != 0 and force is False:
        print_error("B has been initialized. Specify 'force' to initialize W explicitly.")
        return False
    temp = [config.input_dim] + config.hidden_layer_dim_list + [config.output_dim]
    for i in range(config.layer_num):
        bp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (1, temp[i + 1])).astype(dtype=theano.config.floatX)
        Bo.append(theano.shared(bp, name="Bo%d" % i))

"""
################### Initialize Scan ###########################################
"""


def initialize_y_evaluated():
    if parameters is None:
        initialize_param(False)

    # TODO: Haven't known the input parameter of theano.scan, debugging is needed
    global Y_evaluated

    def sub_step(x, a, y, wi, wh, wo, bh, bo):
        a_t = act(T.dot(x, wi) + T.dot(a, wh) + bh)
        y_t = T.dot(a_t, wo) + bo
        return a_t, y_t

    def step(x_seq, t):
        a = theano.shared(numpy.zeros((config.hidden_layer_dim_list[t], config.hidden_layer_dim_list[t])))
        _y_0 = theano.shared(numpy.zeros((config.batch_num, config.hidden_layer_dim_list[t])))
        [_a_seq, _y_seq], _ = theano.scan(
            sub_step,
            sequences=x_seq,
            outputs_info=[a, _y_0],
            non_sequences=[Wi[t], Wh[t], Wo[t], Bh[t], Bo[t]],
            truncate_gradient=-1
        )
        return _y_seq

    i = theano.shared(0)
    y_seq, _ = theano.scan(
        step,
        outputs_info=X,
        sequences=T.arange(0, config.layer_num),
        truncate_gradient=-1
                )
    Y_evaluated = y_seq[-1]


def initialize_scan():
    # Note that we have ensured that all input have the same length
    # Note that this function can only be called after all initialization have been done
    initialize_step_a()
    initialize_step_y()


def initialize_cost():
    if Y_evaluated is None:
        initialize_y_evaluated()

    global cost, get_cost
    cost = cost_function.cost(Y_evaluated, Y)
    get_cost = theano.function(inputs=[X],
                               outputs=cost
                               )


def initialize_grad():
    if cost is None:
        initialize_cost()
    for i in range(len(cost)):
        grad.append(T.grad(cost[i], parameters))
