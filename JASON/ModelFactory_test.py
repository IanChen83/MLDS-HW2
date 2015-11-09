import theano.tensor as tensor
from theano import shared, function, grad
import numpy as np
import theano
import random



class ModelFactory_test:
    def __init__(self, _i_dim, _o_dim, _layer_neuron_num_list=None, _num_batch=1, _lr=0.5):
        # Deal with input parameter
        self.input_dim = _i_dim
        self.output_dim = _o_dim
        self.layer_neuron_num_list = _layer_neuron_num_list
        self.batch_num = _num_batch
        self.learning_rate = _lr
        self.y_evaluated = None
        self.y_evaluated_output = None
        self.cost = None
        self.update_v = None
        self.update = None
        #self.update_lr = None
        self.y_evaluated_function = None
        self.x_input = tensor.fmatrix(name="X_input").astype(dtype = theano.config.floatX)
        self.y_input = tensor.fmatrix(name="Y_input").astype(dtype = theano.config.floatX)
        self.W_array = []
        self.B_array = []
        self.vW_array = []
        self.vB_array = []
        self.random_mu = 0
        self.random_sigma = 0.1
        self.update_momentum = 0.9


        # Deal with class initialization
        self._load_w_array()
        self._create_model()
        self._define_update_function()

    '''
        (Internal) Use layer_neuron_num_list to initialize W_array
    '''

    def _load_w_array(self):
        print "Load W array:\n\t%s\n\t(total %s hidden layer(s))" % (
            self.layer_neuron_num_list,
            len(self.layer_neuron_num_list)
        )
        temp = [self.input_dim] + self.layer_neuron_num_list + [self.output_dim]
        #bp_temp = np.matrix
        for i in range(len(temp) - 1):
            wp = np.random.normal(self.random_mu, self.random_sigma, (temp[i], temp[i + 1])).astype(dtype = theano.config.floatX)
            bp = np.random.normal(self.random_mu, self.random_sigma, (1, temp[i + 1])).astype(dtype = theano.config.floatX)
            #wp = np.random.randn( temp[i], temp[i + 1])
            #bp = np.random.randn(1, temp[i + 1])
            bp = np.tile(bp,(self.batch_num, 1))#.astype(dtype = theano.config.floatX)
            vw = np.zeros((temp[i], temp[i + 1])).astype(dtype = theano.config.floatX)
            vb = np.zeros((self.batch_num, temp[i + 1])).astype(dtype = theano.config.floatX)
            # TODO: W and b are set to zeros
            self.W_array.append(shared(wp, name="W%d" % i))
            self.B_array.append(shared(bp, name="B%d" % i))
            self.vW_array.append(shared(vw, name="vW%d" % i))
            self.vB_array.append(shared(vb, name="vB%d" % i))


    def _create_model(self):
        result = self.x_input
        result2 = self.x_input
        for i in range(len(self.W_array)):
            result = ModelFactory_test._layer_propagate(result, self.W_array[i], self.B_array[i])
        for i in range(len(self.W_array)):
            result2 = ModelFactory_test._layer_propagate(result2, self.W_array[i], self.B_array[i][0])
        self.y_evaluated = result
        self.y_evaluated_output = result2
        # self.y_evaluated_function = function([self.x_input, self.y_input], )

    def _define_update_function(self):
        
        #self.cost = ModelFactory._cost_function(self.y_evaluated, self.y_input)
        #self.cost = -tensor.log(tensor.dot(self.y_evaluated.T,self.y_input)).trace()/self.batch_num 
        #self.cost = tensor.mean((self.y_evaluated-self.y_input).norm(2,axis=1))#-(tensor.log(tensor.dot(self.y_evaluated.T,self.y_input))).trace()
        self.cost = (abs(self.y_evaluated-self.y_input)**2).sum()
        g = tensor.grad(self.cost, self.W_array + self.B_array)
        update_pairs_v = []
        update_pairs = []
        j = len(self.W_array)
        for i in range(len(self.B_array)):
            '''
            update_pairs.append((self.W_array[i], self.W_array[i] - self.learning_rate * g[i]    / self.batch_num))
            update_pairs.append((self.B_array[i], self.B_array[i] - self.learning_rate * g[i + j] / self.batch_num))
            '''
            update_pairs_v.append((self.vB_array[i], self.update_momentum * self.vB_array[i] -
                                 self.learning_rate * g[i + j]/self.batch_num))
            update_pairs_v.append((self.vW_array[i], self.update_momentum * self.vW_array[i] -
                                 self.learning_rate * g[i]/self.batch_num ))
            update_pairs.append((self.W_array[i], self.W_array[i] + self.vW_array[i]))
            update_pairs.append((self.B_array[i], self.B_array[i] + self.vB_array[i]))
            
        #update_pairs.append( (self.learning_rate,self.learning_rate*0.999) )
            
        self.y_evaluated_function = function([self.x_input, self.y_input], self.y_evaluated_output, 
                                             allow_input_downcast=True, on_unused_input='ignore')

        self.cost_function = function([self.x_input, self.y_input], self.cost,allow_input_downcast=True, on_unused_input='ignore')

        self.update_v = function([self.x_input, self.y_input], g, updates=update_pairs_v, allow_input_downcast=True)
        self.update = function([self.x_input, self.y_input], g, updates=update_pairs,
                               allow_input_downcast=True)
        # print dir(self.y_evaluated)

    @staticmethod
    def _act_function(x):
        return 1/(1+tensor.exp(-x))
        #return (1 + tensor.tanh(x / 2)) / 2

    @staticmethod
    def _layer_propagate(layer_input, w, b):
        return ModelFactory_test._act_function(tensor.dot(layer_input, w) + b)

    @staticmethod
    def _cost_function(func, out):
        return (abs(func-out) ** 2).sum()

    def train_one(self, x_input, y_input):
        self.update_v(x_input, y_input)
        return self.update(x_input, y_input)

    def load_parm_4(self, w_input, b_input,vW_input,vB_input):
        temp = [self.input_dim] + self.layer_neuron_num_list + [self.output_dim]
        for i in range(len(temp) - 1):
            wp = w_input[i].get_value()
            bp = b_input[i].get_value()
            vwp = vW_input[i].get_value()
            vbp = vB_input[i].get_value()
            self.W_array[i].set_value(wp)
            self.B_array[i].set_value(bp)
            self.vW_array[i].set_value(vwp)
            self.vB_array[i].set_value(vbp)

    def load_parm_2(self, w_input, b_input):
        temp = [self.input_dim] + self.layer_neuron_num_list + [self.output_dim]
        for i in range(len(temp) - 1):
            wp = w_input[i].get_value()
            bp = b_input[i].get_value()
            self.W_array[i].set_value(wp)
            self.B_array[i].set_value(bp)

    def lr_decade(self):
        self.learning_rate = self.learning_rate*0.999
