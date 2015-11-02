from random import randrange
__author__ = 'patrickchen'


class BColors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        pass


def get_correctness_ratio(correct, total):
    return correct / total


def float_convert(num):
    try:
        return float(num)
    except ValueError:
        return None


def print_data(title, content=None, switch=True):
    if switch is False:
        return
    if content is None:
        print(BColors.GREEN + "* " + str(title) + ":" + BColors.END)
        return
    print(BColors.GREEN + "* " + str(title) + ":" + BColors.END + "\n" + str(content))


def print_error(content):
    print(BColors.RED + "* ERROR " + str(content) + BColors.END)


def my_print(title, content=None, switch=True):
    if switch is False:
        return
    if content is None:
        print(BColors.GREEN + "* " + str(title) + ":" + BColors.END)
        return
    print(BColors.GREEN + "* " + str(title) + ":" + BColors.END + str(content))


def trim_length(seq, length):
    a = randrange(0, len(seq) - length)
    return [s[a, a + length] for s in seq]


def translate_ans(t):
    pass
# try:
#     map_file = open('48_39.map', 'r')
#     in_48 = []
#     in_39 = []
#     for line in map_file:
#         in_x = line.split('\t')
#         in_48.append(in_x[0])
#         in_39.append(in_x[1])
# except IOError:
#     print_error("48_39.map not found")
#
#
# def mapping_max(z, ans_type=0):
#     y = z.tolist()
#     big_index = y.index(max(y))
#     return in_48[big_index] if ans_type == 0 else in_39[big_index]
