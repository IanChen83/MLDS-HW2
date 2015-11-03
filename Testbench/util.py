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


ans_types = [
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d",
            "dh", "dx", "eh", "el", "en", "epi", "er", "ey", "f", "g", "hh",
            "ih", "ix", "iy", "jh", "k", "l", "m", "ng", "n", "ow", "oy", "p",
            "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w",
            "y", "zh", "z"
        ]


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


'''
################## Convert 48 to 39 ##########################################
'''

convert_48_to_39 = {}

try:
    map_file = open('48_39.map', 'r')
    for line in map_file:
        in_x = line.split()

        convert_48_to_39[in_x[0]] = in_x[1]

except IOError:
    print_error("48_39.map not found")


def translate_to_ans_48(t):
    ret = []
    for i in range(len(t)):
        c = [0] * 48
        c[ans_types.index(t[i])] = 1
        ret.append(c)

    return ret


# Use this function to translate test data to output_48 type
def map_max_to_type(z):
    return [ans_types[j.index(max(j))] for j in z]


def get_correctness_num(x, y):
    """
    :param x: input batch
    :return: num of correct
    """
    pass
