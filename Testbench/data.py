try:
    import cPickle
except ImportError:
    import pickle as cPickle
from random import randrange
from util import print_error, translate_to_ans_48
from RNN.config import output_dim
import config
__author__ = 'patrickchen'


training_input = []
training_answer = []
test_input = []
training_input_len = 0


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

        _13 = _11[0].split('_')
        if _13[0] == pre and _13[1] == cen:
            training_input[i].append(_12)
            # Trade-off: Should we save the memory to calculate answers every time?
            # It costs 200 MB to store these data QQ
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


def make_input(ret_no, length):
    """
    :param ret_no: a list containing (No. of seq in training_input, length of the seq)
    :param length: desired length
    :return: (X, Y) to be passed to training function
    """
    ret = [], []
    for j in ret_no:
        a = randrange(0, j[1] - length) if j[1] - length > 0 else 0
        ret[0].append(training_input[j[0]][a:a + length])
        ret[1].append(training_answer[j[0]][a:a + length])
    return ret[0], [translate_to_ans_48(j) for j in ret[1]]


def training_input_random_selection(batch_size, start=0, stop=-1):
    """
        An implementation to generate training input with the same length
    """
    if stop == -1:
        stop = training_input_len
    ret_no = []

    # ret_no is a list containing (no. of seq in training_input, length of the seq)
    for i in range(batch_size):
        x = randrange(start, stop)
        ret_no.append(
            (x, len(training_input[x]))
        )

    x = min(t[1] for t in ret_no)
    return make_input(ret_no, x)


def training_input_sequential_selection(batch_size, start):
    if start + batch_size > training_input_len:
        start = training_input_len - batch_size
        if start < 0:
            print_error("Error: training data < batch size")
            return None

    # ret_no is a list containing (no. of seq in training_input, length of the seq)
    ret_no = [(i, len(training_input[i])) for i in range(start, start+batch_size)]
    x = min(t[1] for t in ret_no)
    return make_input(ret_no, x)
