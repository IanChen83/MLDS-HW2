from util import print_error
from commandline import command_queue
__author__ = 'patrickchen'


def load_script(param):
    if len(param) == 0:
        print_error("ERROR: script command need 1 parameter")
        return False
    try:
        open_file = open(name=param[1], mode='r')
        for command in open_file:
            command_queue.append(command)
        return True
    except IOError:
        print_error("ERROR: script file %s cannot be open" % param[1])
    return False
