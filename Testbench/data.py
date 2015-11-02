import cPickle
from random import randrange
from util import trim_length
__author__ = 'patrickchen'

training_input = []
training_answer = []
test_input = []

ans_type = [
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d",
            "dh", "dx", "eh", "el", "en", "epi", "er", "ey", "f", "g", "hh",
            "ih", "ix", "iy", "jh", "k", "l", "m", "ng", "n", "ow", "oy", "p",
            "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w",
            "y", "zh", "z"
        ]


def load_training_input_raw(filename="train.ark"):
    x = open(filename, 'r')
    for line in x:
        _1 = line.split()
        _2 = [float(i) for i in _1[1:]]
        training_input.append(_2)


def load_training_input(filename="train.ark.cpickle"):
    global training_input

    input_c = file(filename, 'rb')
    training_input = cPickle.load(input_c)
    pass


def write_training_input(filename="train.ark.cpickle"):
    f = file(filename, 'wb')
    cPickle.dump(training_input, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def make_input(batch_size):
    """
        An implementation to generate training input with the same length
    """
    len_train = len(training_input)
    ret_no = []

    for i in range(batch_size):
        x = randrange(0, len_train)
        ret_no.append(
            (x, len(training_input[x]))
        )

    x = min(len(t[1]) for t in ret_no)
    return [trim_length(training_input[i], x) for i in range(batch_size)]

