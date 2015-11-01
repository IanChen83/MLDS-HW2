import param
__author__ = 'patrickchen'


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


def cost(y_e, y):
    return norm_2(y_e, y)
