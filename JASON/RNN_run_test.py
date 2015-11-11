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
batch_num = 10
#sentence max length
len_max = 200


x_seq = T.ftensor3()
y_hat = T.ftensor3()
mask = T.ftensor3()
start = T.scalar()
PARM = T.matrix()

#################### LOAD PARAMETER #################
parm_data = file('parameter_RNN_batch_1111_1.txt','rb')
parm = cPickle.load(parm_data)
Wi = parm[0].get_value() 
bh = parm[1].get_value() #need mean
Wo = parm[2].get_value() 
bo = parm[3].get_value()
Wh = parm[4].get_value() 

print Wi
print bh
print Wo
print bo
print Wh

Wi = theano.shared(Wi)
bh = theano.shared(bh)
Wo= theano.shared(Wo)
bo = theano.shared(bo)
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

test_f = open('DNN_softmax_test.txt','r')
test_ans = open('RNN_test_ans_1111.csv','w')

f_test = []
name = []
for line in test_f:
    input_x = line.split()
    input_x = [float_convert(i) for i in input_x]
    name_x = input_x[0].split('_')
    name.append(name_x)
    f_test.append(input_x)

test_num = len(f_test)

wav_end_test=[]
for i in range( test_num ):
    if i != test_num-1:
        if int(name[ i +1 ][2]) != (int(name[ i ][2])+1) :
            wav_end_test.append((i))
    else:
        wav_end_test.append(i)

wav_end=[]
for i in range( len(name) ):
    if i != len(name)-1:
        if int(name[i][2])+1 != int(name[i+1][2]):
            wav_end.append(i)
    else:
        wav_end.append(i)


c = MAP()
Y=None
m=0
test_ans.write('Id,Prediction\n')


err=0.0
m=0
test_index=0
now_rand_num = 0

now_test_time = 0
wav_count = 0 
end_test = 0

while(end_test==0):
    X=[]
    Y=[]
    count777 = 0
    num = []
    count_len = []
    flag_wav_end = []
    wav_len = []
    this_batch_size = batch_num
    rand_num = 0

    while(rand_num<batch_num):
        if wav_end_test[wav_count]>len_max+m-1:
            num.append(m)
            m = m + len_max
            rand_num = rand_num+1
        else:
            num.append(m)
            if m+len_max<test_num:
                m = wav_end_test[wav_count]+1
            else:
                end_test = 1
                a = [x for x  in num if x!=m]
                this_batch_size = len(a)+1
            if wav_count < len(wav_end_test)-1:
                wav_count = wav_count+1
            rand_num = rand_num+1
        count_len.append( 0 ) 
        flag_wav_end.append(0)
        wav_len.append(0)

    #print 'num',num

    while(count777<len_max):
        X_batch = []
        Y_batch = []
        for bi in range(batch_num):
            now_i = num[bi]
            if( (count_len[bi]+now_i) in wav_end):
                if flag_wav_end[bi]==0:
                    flag_wav_end[bi] = 1
                    if(name[count_len[bi]+now_i][0][0] == 'f'):
                        X_batch.append(np.array([0]+f_test[count_len[bi]+now_i][1:49]).astype(dtype = theano.config.floatX))
                    else:
                        X_batch.append(np.array([1]+f_test[count_len[bi]+now_i][1:49]).astype(dtype = theano.config.floatX))
                    wav_len[bi] = (count_len[bi]+1-1)%len_max +1
                    #count_len[bi] = count_len[bi]+1
                else:
                    yy = [0]*49
                    X_batch.append(yy)
            else:
                if(name[count_len[bi]+now_i][0][0] == 'f'):
                    X_batch.append(np.array([0]+f_test[count_len[bi]+now_i][1:49]).astype(dtype = theano.config.floatX))
                else:
                    X_batch.append(np.array([1]+f_test[count_len[bi]+now_i][1:49]).astype(dtype = theano.config.floatX))
                wav_len[bi] = (count_len[bi]+1-1)%len_max +1
                count_len[bi] = count_len[bi]+1

        
        X.append(X_batch)
        Y.append(Y_batch)
        count777 = count777+1
    Ya = rnn_test_y_evaluate(X)       
    #print 'wav_len',wav_len
    if (len(X)!=len_max):
        print 'len wrong!!'
    Ya_new = []
    for index_i in range(this_batch_size):
        Ya_new_temp = []
        for index in range(len_max):
            Ya_new_temp.append (Ya[index][index_i])
        Ya_new.append(Ya_new_temp)
    #temp = -1
    for index in range(this_batch_size):
        now_i = num[index]

        #print 'wav_len[index]',wav_len[index]
        for index_b in range( wav_len[index] ):
            
            test_ans.write(f_test[now_i+index_b][0])
            test_ans.write(',')
            test_ans.write(c.map(Ya_new[index][index_b],1).split('\n')[0] )
            #print 'index',index
            #print 'index_b',index_b
            test_ans.write('\n')
            #if temp+1!=now_i+index_b:
            #    print 'here!!!!!!!!!!',now_i+index_b
            #    print 'wav_len',wav_len[index]
            #print now_i+index_b
            temp = now_i+index_b
