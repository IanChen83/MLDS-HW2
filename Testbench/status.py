from util import my_print
import RNN.config as rc
import config as tc
__author__ = 'patrickchen'


def print_status(param):
    if len(param) == 0 or param[0] == "testbench":
        print "========== TestBench Status =========="
        my_print("Train input file", tc.train_input_file)
        my_print("Train answer file", tc.train_answer_file)
        my_print("Test input file", tc.test_input_file)
        my_print("Train input file", tc.test_output_file)

    if len(param) == 0 or param[0] == "rnn":
        print "========== RNN Status ================"
        my_print("Input dimension", rc.input_dim)
        my_print("Output dimension", rc.output_dim)
        my_print("Hidden layer list", rc.hidden_layer_dim_list)
        my_print("Batch size", rc.batch_num)
        my_print("Learning rate", rc.learning_rate)

    print "======================================"
