import sys

import commandline
import data
import RNN.param
import RNN.train
import train
import status

__author__ = 'patrickchen'

argv = None


def main():
    # Print status?
    init()
    # Invoke command line
    commandline.run()


def init():
    commandline.register_command("status", status.print_status)
    commandline.register_command("cost", train.get_acc)
    commandline.register_command("exit", exit)

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

    print("@ Initialize train")
    RNN.train.initialize_train()
    print("@ Initialize train FINISH")

    train.get_acc(None)

if __name__ == "__main__":
    argv = sys.argv[1:]
    main()
else:
    print("Invoke 'test.py' directly. About")
    exit(2)
