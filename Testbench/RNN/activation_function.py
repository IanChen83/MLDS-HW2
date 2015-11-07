from theano import tensor
__author__ = 'patrickchen'


def sigmoid(func):
    """
    :param func:    tensor.TensorVariable
    :return:        tensor.TensorVariable
    """
    return (1 + tensor.tanh(func / 2)) / 2


def relu(func):
    """
    :param func:    tensor.TensorVariable
    :return:        tensor.TensorVariable
    """

    # return a sufficient small positive number and a sufficient big positive number
    return func.clip(0.00000001, 1000000)


def act(func):
    return sigmoid(func)
