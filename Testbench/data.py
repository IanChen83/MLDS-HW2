try:
    import cPickle
except ImportError:
    import pickle as cPickle
from random import randrange
from util import trim_length, print_error
from RNN.config import output_dim
import config
__author__ = 'patrickchen'


training_input = []
training_answer = []
test_input = []
training_input_len = 0

ans_type = [
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d",
            "dh", "dx", "eh", "el", "en", "epi", "er", "ey", "f", "g", "hh",
            "ih", "ix", "iy", "jh", "k", "l", "m", "ng", "n", "ow", "oy", "p",
            "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w",
            "y", "zh", "z"
        ]


def load_training_data_raw(input_data=config.test_input_file, ans_data=config.train_answer_file):
    global training_input_len
    x = [_x.rstrip() for _x in open(input_data, 'r')]
    y = [_y.rstrip() for _y in open(ans_data, 'r')]
    i = -1
    pre = ""
    cen = ""
    _n = len(x)
    for s in range(_n):
        input_line = x[s]
        ans_line = y[s]
        _11 = input_line.split()
        _12 = [float(t) for t in _11[1:]]

        _21 = ans_line.split(',')
        _22 = [0] * output_dim
        _22[ans_type.index(_21[1])] = 1

        _13 = _11[0].split('_')
        if _13[0] == pre and _13[1] == cen:
            training_input[i].append(_12)
            training_answer[i].append(_21[1])
        else:
            pre = _13[0]
            cen = _13[1]
            training_input.append([])
            training_answer.append([])
            i += 1
            training_input[i].append(_12)
            training_answer[i].append(_21[1])

    training_input_len = len(training_input)


def load_training_input(filename="train.ark.cpickle"):
    global training_input, training_answer, training_input_len

    input_c = open(filename, 'rb')
    training_input, training_answer = cPickle.load(input_c)
    training_input_len = len(training_input)


def write_training_input(filename="train.ark.cpickle"):
    f = open(filename, 'wb')
    cPickle.dump((training_input, training_answer), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def make_training_input_random(batch_size, start=0, stop=-1):
    """
        An implementation to generate training input with the same length
    """
    if stop == -1:
        stop = training_input_len
    ret_no = []

    for i in range(batch_size):
        x = randrange(start, stop)
        ret_no.append(
            (x, len(training_input[x]))
        )

    x = min(t[1] for t in ret_no)
    ret_x = []
    ret_y = []
    for i in range(batch_size):
        a = randrange(0, training_input[i] - x)
        ret_x.append(
            training_input[ret_no[i][0]][a:a+x]
        )
        ret_y.append(
            training_answer[ret_no[i][0]][a:a+x]
        )
    return ret_x, ret_y


def make_training_input_sequential(batch_size, start):
    if start + batch_size > training_input_len:
        start = training_input_len - batch_size
        if start < 0:
            print_error("Error: training data < batch size")
            return None
    ret_no = [(i, len(training_input[i])) for i in range(start, start+batch_size)]
    x = min(t[1] for t in ret_no)
    ret_x = []
    ret_y = []
    for i in range(batch_size):
        a = randrange(0, training_input[i] - x)
        ret_x.append(
            training_input[ret_no[i][0]][a:a+x]
        )
        ret_y.append(
            training_answer[ret_no[i][0]][a:a+x]
        )
    return ret_x, ret_y
