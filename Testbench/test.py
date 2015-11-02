import sys
import config

import command.status
import command.script
import commandline
import data
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

    print "===== Import training data ==========="
    data.load_training_input_raw("train_sub.ark")
    print "===== Import training data Finish ===="


if __name__ == "__main__":
    argv = sys.argv[1:]
    main()
else:
    print "Invoke 'test.py' directly. About"
    exit(2)
