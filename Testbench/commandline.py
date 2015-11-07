from collections import deque

from util import BColors, print_error
__author__ = 'patrickchen'

command_queue = deque()

prompt = "Enter a command:"

dispatcher = {}

try:
    input = raw_input
except NameError:
    pass

def __do_nothing__(_):
    pass


def run():
    global prompt
    while True:
        command = raw_input(BColors.BLUE + prompt + BColors.END)
        command_queue.append(command)

        if len(command_queue) != 0:
            __exec_one_command__(command_queue.popleft())

        prompt = "Enter a command:"


def __exec_one_command__(cmd):
    commands = cmd.strip().split()
    if len(commands) == 0:
        return
    func = dispatcher.get(commands[0], lambda _: __do_nothing__)
    func(commands[1:])


def register_command(word, func):
    if word in dispatcher:
        print_error("Register command warning: func will be replaced because key already exists")
    dispatcher[word] = func
