import RNN.config as config
import RNN.param as param

__author__ = 'patrickchen'

update = None


def initialize_update():
    global update
    update = [(p, p - config.learning_rate * g) for p, g in zip(param.parameters, param.grad)]
