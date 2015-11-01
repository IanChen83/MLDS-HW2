import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
from  output48_39 import *
__author__= 'jason'

# Min/max sequence length
#MIN_LENGTH = 50
#MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 128
# input
N_INPUT = 48
# output
N_OUTPUT = 48

x_seq = T.matrix()
y_hat = T.matrix()
mask = T.vector()
y_seq_modify = T.matrix()

Wi = theano.shared( np.random.randn(N_INPUT,N_HIDDEN) )
bh = theano.shared( np.zeros(N_HIDDEN) )
Wo = theano.shared( np.random.randn(N_HIDDEN,N_OUTPUT) )
bo = theano.shared( np.zeros(N_OUTPUT) )
Wh = theano.shared( np.identity(N_HIDDEN) )
parameters = [Wi,bh,Wo,bo,Wh]

def sigmoid(z):
        return 1/(1+T.exp(-z))

def step(x_t,a_tm1,y_tm1):
        a_t = sigmoid( T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh )
        y_t = T.dot(a_t,Wo) + bo
        return a_t, y_t

a_0 = theano.shared(np.zeros(N_HIDDEN))
y_0 = theano.shared(np.zeros(N_OUTPUT))

[a_seq,y_seq],_ = theano.scan(
                        step,
                        sequences = x_seq,
                        outputs_info = [ a_0, y_0 ],
                        truncate_gradient=-1
                )

#y_seq_last = y_seq[-1][0] # we only care about the last output 
y_seq_modify = (y_seq.T*mask).T 
cost = T.sum( ( y_seq_modify - y_hat )**2 ) 

gradients = T.grad(cost,parameters)


def MyUpdate(parameters,gradients):
	mu =  np.float32(0.001)
	parameters_updates = [(p,p - mu * T.clip(g,-10,10)) for p,g in izip(parameters,gradients) ] 
	return parameters_updates

rnn_test_cost = theano.function(
        inputs= [x_seq,y_hat,mask],
        outputs = cost
        #allow_input_downcast=True, on_unused_input='ignore'
)

rnn_test_y_evaluate = theano.function(
        inputs= [x_seq,mask],
        outputs = y_seq_modify
        #allow_input_downcast=True, on_unused_input='ignore'
)

rnn_train = theano.function(
        inputs=[x_seq,y_hat,mask],
        outputs=cost,
	updates=MyUpdate(parameters,gradients)
)

def float_convert(i):
    try: 
        return np.float32(i)
    except ValueError :
        return i

######################################################## testbench part ##########################################
f = open('DNN.txt','r')
ans_data = open('answer_map.txt','r')

f_DNNsoft = []
name = []
for line in f:
    input_x = line.split()
    input_x = [float_convert(i) for i in input_x]
    name_x = input_x[0].split('_')
    name.append(name_x)
    f_DNNsoft.append(input_x)
ans = []
for line in ans_data:
    ans_x = line.split(',')
    ans.append(ans_x[1])


anstype = ["aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d", "dh", "dx", "eh", "el"
               , "en", "epi", "er", "ey", "f", "g", "hh", "ih", "ix", "iy", "jh", "k", "l", "m", "ng"
               , "n", "ow", "oy", "p", "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w"
               , "y", "zh", "z"]

train_number = 1091215 #1124823
validation_num = 1091422-1091215
c = MAP()
ACC = 0
counter = 0
epoch=0
mask = []
try:
    while True:
        X = []
        Y = []
        count777 = 0
        flag_wav_end = 0
        flag_data_end = 0
        counter =counter+1
        wav_len = 0
        if i>=train_number:#i>=1124823:
            i=0
            epoch=1
        if counter%1000 == 0:
            print i
        while (count777<777) :
            if(i==train_number-1):
                if(flag_data_end==0):
                    typeidx = anstype.index(str(ans[i].split('\n')[0]))
                    y=[0]*48
                    y[typeidx]=1
                    Y.append(y)
                    X.append(f_DNNsoft[i][1:49])
                    flag_data_end = 1
                    wav_len = int(name[i][2])
                else:
                    y=[0]*48
                    Y.append(y)
                    X.append(y)
            else:
                if(name[i][0]==name[i+1][0] and name[i][1]==name[i+1][1]) :
                    typeidx = anstype.index(str(ans[i].split('\n')[0]))
                    y=[0]*48
                    y[typeidx]=1
                    Y.append(y)
                    X.append(f_DNNsoft[i][1:49])
                    i=i+1
                else: 
                    if(flag_wav_end==0):
                        typeidx = anstype.index(str(ans[i].split('\n')[0]))
                        y=[0]*48
                        y[typeidx]=1
                        Y.append(y)
                        X.append(f_DNNsoft[i][1:49])
                        flag_wav_end = 1
                        wav_len = int(name[i][2])
                    else:
                        y=[0]*48
                        Y.append(y)
                        X.append(y)
            count777 = count777+1;
        i=i+1

        mask = np.ones(wav_len).tolist()+np.zeros(777-wav_len).tolist()
        rnn_train(X,Y,mask)

        if epoch==1:
            #print rnn_test_cost(X,Y,mask)
            err=0.0
            m=0
            while(m<validation_num):
                X_test=[]
                Y_test=[]
                flag_data_end_test = 0
                flag_wav_end_test = 0
                count777_test = 0
                wave_lengh = 0;
                while (count777_test<777) :
                    if(m==validation_num-1):
                        if(flag_data_end_test==0):
                            typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                            y=[0]*48
                            y[typeidx]=1
                            Y_test.append(y)
                            X_test.append(f_DNNsoft[train_number+m][1:49])
                            flag_data_end_test = 1
                            wave_lengh = int(name[train_number+m][2])
                        else:
                            y=[0]*48
                            Y_test.append(y)
                            X_test.append(y)
                    else:
                        if(name[train_number+m][0]==name[train_number+m+1][0] and name[train_number+m][1]==name[train_number+m+1][1]) :
                            typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                            y=[0]*48
                            y[typeidx]=1
                            Y_test.append(y)
                            X_test.append(f_DNNsoft[train_number+m][1:49])
                            m=m+1
                        else: 
                            if(flag_wav_end_test==0):
                                typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                                y=[0]*48
                                y[typeidx]=1
                                Y_test.append(y)
                                X_test.append(f_DNNsoft[i][1:49])
                                flag_wav_end_test = 1
                                wave_lengh = int(name[train_number+m][2])
                            else:
                                y=[0]*48
                                Y_test.append(y)
                                X_test.append(y)
                    count777_test = count777_test+1;
                m=m+1
                mask_test = np.ones(wave_lengh).tolist()+np.zeros(777-wave_lengh).tolist()
                Ya = rnn_test_y_evaluate(X,mask_test)
                #print "wave_lengh",wave_lengh
                for index in range(wave_lengh):
                    if( c.map(Ya[index]) !=  str(ans[train_number+m-wave_lengh+index].split('\n')[0]) ):
                        err = err+1
                        print "y_evaluate",Ya[index] 
                        print train_number+m-wave_lengh+index
                        print "test_name",name[train_number+m-wave_lengh+index][0],name[train_number+m-wave_lengh+index][1],"ANS",c.map(Ya[index])
                        print [str(ans[train_number+m-wave_lengh+index].split('\n')[0])]
                #print m
            ACC = 1.0-err/validation_num
            print 'ACC = %f'%(ACC)
            epoch = 0
            counter = 0
except KeyboardInterrupt:
    pass


f.close()
ans_data.close()


'''
for i in range(1):
	x_seq, y_hat = gen_data()
	print "iteration:", i, "cost:",  rnn_train(x_seq,y_hat)

for i in range(1):
	x_seq, y_hat = gen_data()
	print "reference", y_hat, "RNN output:", rnn_test(x_seq)
'''