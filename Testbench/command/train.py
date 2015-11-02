import Testbench.config as config
import Testbench.data as data
from Testbench.util import print_error, my_print
import RNN.config
import RNN.train
from RNN.train import initialize_train
__author__ = 'patrickchen'


def run(param):
    pass


def get_acc(param):
    if len(data.training_input) < config.training_segment:
        print_error("ERROR: training segment > data number")
        return False
    i = config.training_segment
    while i < data.training_input_len:
        d = RNN.train.cost(*data.make_training_input_sequential(RNN.config.batch_num, i))
        my_print("Cost", d)
