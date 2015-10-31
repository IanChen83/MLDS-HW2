from theano import tensor

import cPickle
import config
import theano
import numpy
from util import print_error
import activation_function

__author__ = 'patrickchen'

X = tensor.fmatrix()
Y = tensor.fmatrix()
Y_evaluated = None

# 'i' stands for input
# 'h' stands for history
# 'o' stands for output
Wi = []
Wh = []
Bh = []
Wo = []
Bo = []


def load_param_from_file(filename):
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
    config.hidden_layer_dimension_list = conf.lidden_layer_dimension_num_list
    config.batch_num = conf.batch_num
    config.layer_num = conf.layer_num

    for i in range(config.layer_num):
        Wi[i].set_value(w_param[i].get_value())
        Wh[i].set_value(w_param[i + config.layer_num].get_value())
        Wo[i].set_value(w_param[i + config.layer_num * 2].get_value())
        Bh[i].set_value(b_param[i].get_value())
        Bo[i].set_value(b_param[i + config.layer_num].get_value())

    return True


def write_param(filename):
    conf = config.DumpConfig()
    conf.input_dim = config.input_dim
    conf.output_dim = config.output_dim
    conf.hidden_layer_dimension_list = config.hidden_layer_dimension_list
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
    if initialize_wi() is False\
            or initialize_wh() is False\
            or initialize_wo() is False\
            or initialize_bh() is False\
            or initialize_bo() is False:
        return False
    return True

def initialize_y_evaluated(x):
    result = x
    for i in range(config.layer_num):
        result = activation_function.act(
            tensor.dot(result, Wi[i]) + Bh[i].dimshuffle('x', 1)
        )
    return result


def initialize_wi(force=False):
    if len(Wi) != 0 and force is False:
        print_error("W has been initialized. Specify 'force' to initialize W explicitly.")
        return False

    temp = [config.input_dim] + config.hidden_layer_dimension_list + [config.output_dim]
    for i in range(config.layer_num):
        wp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (temp[i], temp[i + 1])).astype(dtype=theano.config.floatX)
        Wi.append(theano.shared(wp, name="Wi%d" % i))
    return True


def initialize_wh(force=False):
    if len(Wi) != 0 and force is False:
        print_error("W has been initialized. Specify 'force' to initialize W explicitly.")
        return False

    temp = [config.input_dim] + config.hidden_layer_dimension_list + [config.output_dim]
    for i in range(config.layer_num):
        wp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (temp[i + 1], temp[i + 1])).astype(dtype=theano.config.floatX)
        Wh.append(theano.shared(wp, name="Wh%d" % i))
    return True


def initialize_wo(force=False):
    if len(Wi) != 0 and force is False:
        print_error("W has been initialized. Specify 'force' to initialize W explicitly.")
        return False

    temp = [config.input_dim] + config.hidden_layer_dimension_list + [config.output_dim]
    for i in range(config.layer_num):
        wp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (temp[i + 1], temp[i + 1])).astype(dtype=theano.config.floatX)
        Wo.append(theano.shared(wp, name="Wo%d" % i))
    return True


def initialize_bh(force=False):
    if len(Bh) != 0 and force is False:
        print_error("B has been initialized. Specify 'force' to initialize W explicitly.")
        return False
    temp = [config.input_dim] + config.hidden_layer_dimension_list + [config.output_dim]
    for i in range(config.layer_num):
        bp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (1, temp[i + 1])).astype(dtype=theano.config.floatX)
        Bh.append(theano.shared(bp, name="Bh%d" % i))
    return True


def initialize_bo(force=False):
    if len(Bh) != 0 and force is False:
        print_error("B has been initialized. Specify 'force' to initialize W explicitly.")
        return False
    temp = [config.input_dim] + config.hidden_layer_dimension_list + [config.output_dim]
    for i in range(config.layer_num):
        bp = numpy.random.normal(config.random_mu, config.random_sigma,
                                 (1, temp[i + 1])).astype(dtype=theano.config.floatX)
        Bo.append(theano.shared(bp, name="Bo%d" % i))
