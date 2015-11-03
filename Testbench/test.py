import sys

import command.status
import command.script
import command.train
import commandline
import data
import RNN.param
__author__ = 'patrickchen'

argv = None


def main():
    # Print status?
    init()
    # Invoke command line
    commandline.run()


def init():
    commandline.register_command("status", command.status.print_status)
    commandline.register_command("exit", exit)
    commandline.register_command("cost", command.train.get_acc)

    print("@ Import training data")
    data.load_training_data_raw("train_sub.ark", "answer_map_sub.txt")
    print("@ Import training data FINISH")

    print("@ Initialize parameters")
    RNN.param.initialize_param(False)
    print("@ Initialize parameters FINISH")

    print("@ Initialize y_evaluated")
    RNN.param.initialize_y_evaluated()
    print("@ Initialize y_evaluated FINISH")

    print("@ Initialize cost and grad")
    RNN.param.initialize_cost()
    RNN.param.initialize_grad()
    print("@ Initialize cost and grad FINISH")
    command.train.get_acc(None)

if __name__ == "__main__":
    argv = sys.argv[1:]
    main()
else:
    print("Invoke 'test.py' directly. About")
    exit(2)
