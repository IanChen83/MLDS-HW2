import theano.tensor as tensor
from theano import shared, function, grad
import numpy as np
import theano
import random
import cPickle
import pdb
from output48_39 import *

'''
#####data generation#####
'''
def float_convert(i):
    try: 
        return np.float32(i)
    except ValueError :
        return i


train_data = open('DNN.txt','r')
ans_data = open('answer_map.txt','r')
train = []
frame = []
frameans = []
ans = []
anstemp = ans_data.readlines()
temp = train_data.readlines()
speaker = temp[0].split("_")
longframe = 0
count = 1
anstype = ["aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d", "dh", "dx", "eh", "el"
               , "en", "epi", "er", "ey", "f", "g", "hh", "ih", "ix", "iy", "jh", "k", "l", "m", "ng"
               , "n", "ow", "oy", "p", "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w"
               , "y", "zh", "z"]

for kk in range(len(temp)):
    input_x = temp[kk].split()
    ans_x = anstemp[kk]
    if speaker[0] == input_x[0].split("_")[0] and speaker[1] == input_x[0].split("_")[1]:
        input_x = [float_convert(i) for i in input_x]
        frame.append(input_x)
        frameans.append(ans_x.split(","))
        count += 1
    else:
        train.append(frame)
        ans.append(frameans)
        frame = []
        frameans = []
        frame.append(input_x)
        frameans.append(ans_x.split(","))
        speaker = input_x[0].split("_")
        if count > longframe:
            longframe = count
        count = 1









input_dim = 69
output_dim = 48
layer_list = [128]
learning_rate = 0.9
random_mu = 0
random_sigma = 0.1
batch_num = 1
Wh_array = []
vWh_array = []
W_array = []
vW_array = []
B_array = []
vB_array = []
x_seq = T.matrix()
a_0 = theano.shared(np.zeros(layer_list[0]))
'''
for i in range(len(layer_list)):
    a_0.append(theano.shared(np.zeros(layer_list[i])))
'''
y_0 = theano.shared(np.zeros(output_dim))
real = []
this_input = []
update_pairs = []

'''
###########load array#######
'''
print "Load W array:\n\t%s\n\t(total %s hidden layer(s))" % (
    layer_list,
    len(layer_list)
)
temp = [input_dim] + layer_list + [output_dim]
for i in range(len(temp) - 1):
    wp = np.random.normal(random_mu, random_sigma, (temp[i], temp[i + 1])).astype(dtype = theano.config.floatX)
    bp = np.random.normal(random_mu, random_sigma, (1, temp[i + 1])).astype(dtype = theano.config.floatX)
    bp = np.tile(bp,(batch_num, 1))#.astype(dtype = theano.config.floatX)
    #vw = np.zeros((temp[i], temp[i + 1])).astype(dtype = theano.config.floatX)
    #vb = np.zeros((batch_num, temp[i + 1])).astype(dtype = theano.config.floatX)
    if i != 0:
        whp = np.random.normal(random_mu, random_sigma, (temp[i], temp[i])).astype(dtype = theano.config.floatX)
        #vwh = np.zeros((temp[i], temp[i])).astype(dtype = theano.config.floatX)
        Wh_array.append(shared(whp, name="Wh%d" % i))
        #vWh_array.append(shared(vwh, name="vWh%d" % i - 1))

    W_array.append(shared(wp, name="W%d" % i))
    B_array.append(shared(bp, name="B%d" % i))
        #vW_array.append(shared(vw, name="vW%d" % i))
        #vB_array.append(shared(vb, name="vB%d" % i))


def act_function(x):
    return 1/(1+tensor.exp(-x))


def step(x_t, a_tm1, y_tm1):
    y_t = []
    a_t = []
    '''
    a_t.append(act_function(tensor.dot(x_t, W_array[0]) + tensor.dot(a_tm1[0], Wh_array[0]) + B_array[0]))
    for i in range(len(W_array) - 2):
        a_t.append(act_function(tensor.dot(a_t[i], W_array[i + 1]) + tensor.dot(a_tm1[i + 1], Wh_array[i + 1]) + B_array[i + 1]))
    y_t = tensor.dot(a_t[len(a_t) - 1], W_array[len(W_array) - 1]) + B_array[len(B_array) - 1]
    '''
    a_t = act_function(tensor.dot(x_t, W_array[0] + tensor.dot(a_tm1, Wh_array[0]) + B_array[0]))
    y_t = tensor.dot(a_t, W_array[1]) + B_array[1]
    print type(a_t)
    print type(y_t)
    return a_t, y_t


'''
##########scan#########
'''
pdb.set_trace()
[a_seq, y_seq], update = theano.scan(step, sequences = x_seq, outputs_info = [a_0, y_0], truncate_gradient = -1)


'''
##########cost_function######
'''
def cost_function(myestimate, myreal, mythis_input):
    idx = 0
    myfunc = []
    myout = []
    for i in range(len(mythis_input)):
        if mythis_input[i] != [0]*69:
            idx = i
    for i in range(idx + 1):
        myout.append(myestimate[i])
        myfunc.append(myreal[i])
    return ((myfunc-myout) ** 2).sum()


'''
########gradient#######
'''

gradient = tensor.grad(cost_function(y_seq[-1], real, this_input), W_array + B_array + Wh_array)


'''
#######update#######
'''
for i in range(len(B_array)):
    update_pairs.append((W_array[i], W_array[i] - learning_rate * g[i] / batch_num ))
    update_pairs.append((B_array[i], B_array[i] - learning_rate * g[i + len(W_array)] / batch_num))
for i in range(len(self.Wh_array)):
    update_pairs.append((Wh_array[i], Wh_array[i] - learning_rate * g[i + len(W_array) + len(B_array)] / batch_num))


rnn_test = theano.function(inputs = [x_seq], outputs = y_seq[-1])

rnn_train = theano.function(inputs = [x_seq, y_hat], outputs = cost_function(y_seq[-1], real, this_input), updates = update_pairs)


i=0
ACC = 0.0
W_new = []
B_new = [] 
c = MAP()
epoch = 0
try:
    while True:
        X = []
        Y = []
        err = 0.0
        total = 0
        for i in range(len(train)):
            for j in range(len(train[i])):
                typeidx = anstype.index(str(ans[i][j][1].split('\n')[0]))
                y=[0]*48
                y[typeidx]=1
                Y.append(y)
                X.append(train[i][j][1:70])
                for ii in range(count - len(train[i])):
                    Y.append([0]*48)
                    X.append([0]*69)
                y_hat = np.array(Y)
                x_seq = np.array(X)
                if i >= range(len(train) - 500):
                    estimate = rnn_test(x_seq)
                    idx = 0
                    for i in range(len(X)):
                        if X[i] != [0]*69:
                            idx = i
                    for i in range(idx+1):
                        if [c.map(estimate[idx])]!=[str(ans[i][idx].split('\n')[0])]:
                            err = err+1
                            total += idx
                else:
                    rnn_train(x_seq, y_hat)
        ACC = 1.0-err/total
        print 'ACC = %f'%(ACC)


except KeyboardInterrupt:
    pass

