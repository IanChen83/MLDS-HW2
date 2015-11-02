__author__ = 'patrickchen'

input_dim = 0
output_dim = 0
hidden_layer_dim_list = []
batch_num = 1
layer_num = 1

random_mu = 0
random_sigma = 0.1

learning_rate = 0.001


class DumpConfig:
    def __init__(self):
        self.input_dim = 0
        self.output_dim = 0
        self.hidden_layer_neuron_num_list = []
        self.batch_num = 1
        self.layer_num = 1


def output_shape():
    return 1, batch_num
