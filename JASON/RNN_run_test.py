import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
import cPickle
from  output48_39 import *
from random import randrange
import pdb
__author__= 'JasonWu'

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 128
# input
N_INPUT = 49 # 48 + 1 (male = 1, female = 0)
# output
N_OUTPUT = 48
#mini batch
batch_num = 1
#sentence max length
len_max = 777

x_seq = T.matrix()
y_hat = T.matrix()
mask = T.matrix()
start = T.scalar()
PARM = T.matrix()

#################### LOAD PARAMETER #################
parm_data = file('parameter_RNN_1108_1.txt','rb')
parm = cPickle.load(parm_data)
Wi = parm[0].get_value() 
bh = parm[1].get_value()
Wo = parm[2].get_value() 
bo = parm[3].get_value()
Wh = parm[4].get_value() 

print Wi
print bh
print Wo
print bo
print Wh

Wi = theano.shared(Wi)
bh = theano.shared(bh[0])
Wo= theano.shared(Wo)
bo = theano.shared(bo[0])
Wh = theano.shared(Wh)

#sigma = theano.shared(np.random.randn(N_INPUT,N_HIDDEN) )
#Wi = theano.shared( np.random.normal(0, 0.1, (N_INPUT,N_HIDDEN)) )
#Wo = theano.shared( np.random.normal(0, 0.1, (N_HIDDEN,N_OUTPUT)) )
parameters = [Wi,bh,Wo,bo,Wh]

a_0 = np.zeros(N_HIDDEN).astype(dtype = theano.config.floatX)
a_0 = theano.shared( np.tile(a_0,(batch_num,1)) )
y_0 = np.zeros(N_OUTPUT).astype(dtype = theano.config.floatX)
y_0 = theano.shared( np.tile(y_0,(batch_num,1)) )

a_1 = theano.shared( np.zeros(N_HIDDEN) , name='a1')
y_1 = theano.shared( np.zeros(N_HIDDEN) , name='a1')

def sigmoid(z):
    return 1/(1+T.exp(-z))

def step(z_t,a_tm1):
    return sigmoid( z_t+ T.dot(a_tm1,Wh) + bh )

z_seq = T.dot (x_seq, Wi)

a_seq,_ = theano.scan(
                        step,
                        sequences = z_seq,
                        outputs_info =  a_0,  
                        truncate_gradient=-1
                )

y_seq = T.dot(a_seq,Wo)+bo


y_seq_modify =  (y_seq*mask) 
cost = (( y_seq_modify - y_hat )**2 ).sum() / batch_num

#y_seq_modify_1 = (y_seq.T*mask).T 
#y_seq_modify = (T.exp(y_seq_modify_1).T/ T.sum( T.exp(y_seq_modify_1) , axis=1)).T
#cost = -1*((T.log(y_seq_modify)*y_hat).sum())
#y_seq_modify = (y_seq.T*mask).T 
#cost = T.sum( ( y_seq_modify - y_hat )**2 ) 

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        #print acc.get_value()
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * T.clip(g,-10,10) ))
    return updates

rnn_test_cost = theano.function(
        inputs= [x_seq,y_hat,mask],
        outputs = cost
)

rnn_test_y_evaluate = theano.function(
        inputs= [x_seq],
        outputs = y_seq
)

rnn_test_parm = theano.function(
        inputs= [],
        outputs = [Wi,bh,Wo,bo,Wh]
)

rnn_train_test = theano.function(
        inputs=[x_seq,y_hat,mask],
        outputs=cost,
        updates=RMSprop(cost, parameters, lr=0.001, rho=0.9, epsilon=1e-6)
)

def float_convert(i):
    try: 
        return np.float32(i)
    except ValueError :
        return i

######################################################## testbench part ##########################################

######################## test.ark ############################

######################## test ############################

test_f = open('DNN_test_RealBaseLine.txt','r')
test_ans = open('RNN_test_ans_1109.csv','w')

f_test = []
name = []
for line in test_f:
    input_x = line.split()
    input_x = [float_convert(i) for i in input_x]
    name_x = input_x[0].split('_')
    name.append(name_x)
    f_test.append(input_x)

test_num = len(f_test)
test_c = MAP()
Y=None
m=0
test_index=0
test_ans.write('Id,Prediction\n')
while(m<test_num):
    X_test=[]
    Y_test=[]
    flag_data_end_test = 0
    flag_wav_end_test = 0
    count777_test = 0
    wave_lengh = 0;
    while (count777_test<777) :
        if(m==test_num-1):
            if(flag_data_end_test==0):
                #typeidx = anstype.index(str(ans[m].split('\n')[0]))
                #y=[0]*48
                #y[typeidx]=1
                #Y_test.append(y)
                if(name[m][0][0] == 'f'):
                    X_test.append(np.array([0]+f_test[m][1:49]).astype(dtype = theano.config.floatX))
                else:
                    X_test.append(np.array([1]+f_test[m][1:49]).astype(dtype = theano.config.floatX))
                flag_data_end_test = 1
                wave_lengh = int(name[m][2])
            else:
                y=[0]*49
                #Y_test.append(y)
                X_test.append(y)
        else:
            if(name[m][0]==name[m+1][0] and name[m][1]==name[m+1][1]) :
                #typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                #y=[0]*48
                #y[typeidx]=1
                #Y_test.append(y)
                if(name[m][0][0] == 'f'):
                    X_test.append(np.array([0]+f_test[m][1:49]).astype(dtype = theano.config.floatX))
                else:
                    X_test.append(np.array([1]+f_test[m][1:49]).astype(dtype = theano.config.floatX))
                m=m+1
            else: 
                if(flag_wav_end_test==0):
                    #typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                    #y=[0]*48
                    #y[typeidx]=1
                    #Y_test.append(y)
                    if(name[m][0][0] == 'f'):
                        X_test.append(np.array([0]+f_test[m][1:49]).astype(dtype = theano.config.floatX))
                    else:
                        X_test.append(np.array([1]+f_test[m][1:49]).astype(dtype = theano.config.floatX))
                    flag_wav_end_test = 1
                    wave_lengh = int(name[m][2])
                else:
                    y=[0]*49
                    #Y_test.append(y)
                    X_test.append(y)
        count777_test = count777_test+1;
    m=m+1
    Ya = rnn_test_y_evaluate(X_test)
    for index in range(wave_lengh):
        test_ans.write(f_test[test_index+index][0])
        test_ans.write(',')
        test_ans.write(test_c.map(Ya[index]))
        if m!=test_num-1-1:
            test_ans.write('\n')
    test_index = test_index+wave_lengh
