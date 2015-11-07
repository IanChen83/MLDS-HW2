from itertools import izip

import config
import param

__author__ = 'patrickchen'

update = None


def initialize_update():
    global update
    update = [(p, p - config.learning_rate * g) for p, g in izip(param.parameters, param.grad)]
