__author__ = 'patrickchen'

from theano import tensor
import param


def norm_2(func, out):
    """
    :param func:    tensor.TensorVariable
    :param out:     tensor.TensorVariable
    :return:        tensor.TensorVariable
    """
    return (func - out).norm(2, axis=1)


def norm_1(func, out):
    """
    :param func:    tensor.TensorVariable
    :param out:     tensor.TensorVariable
    :return:        tensor.TensorVariable
    """
    return (func - out).norm(1, axis=1)

cost = norm_2(param.Y_evaluated, param.Y)
