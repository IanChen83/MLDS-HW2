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
        print BColors.GREEN, "*", title, ":", BColors.END
        return
    print BColors.GREEN, "*", title, ":", BColors.END, "\n", content


def print_error(content):
    print BColors.RED, "* ERROR ", content, BColors.END


def my_print(title, content=None, switch=True):
    if switch is False:
        return
    if content is None:
        print BColors.GREEN, "*", title, ":", BColors.END
        return
    print BColors.GREEN, "*", title, ":", BColors.END, content


try:
    map_file = open('48_39.map', 'r')
    in_48 = []
    in_39 = []
    for line in map_file:
        in_x = line.split('\t')
        in_48.append(in_x[0])
        in_39.append(in_x[1])
except IOError:
    print_error("48_39.map not found")


def mapping_max(z, ans_type=0):
    y = z.tolist()
    big_index = y.index(max(y))
    return in_48[big_index] if ans_type == 0 else in_39[big_index]
