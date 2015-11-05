import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
import cPickle
from  output48_39 import *
__author__= 'JasonWu'

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

#y_seq_modify = (T.exp(y_seq).T/ T.sum( T.exp(y_seq) , axis=1)).T
y_seq_modify = (y_seq.T*mask).T
cost = T.sum( ( y_seq_modify - y_hat )**2 )
#cost = -1*((T.log(y_seq_modify)*y_hat).sum())
decay_rate = 0.99
gradients = T.grad(cost,parameters)


def MyUpdate(parameters,gradients):
	mu =  np.float32(0.00001)
	parameters_updates = [(p,p - mu * T.clip(g,-10,10)) for p,g in izip(parameters,gradients) ]
	return parameters_updates

rnn_test_cost = theano.function(
        inputs= [x_seq,y_hat,mask],
        outputs = cost
        #allow_input_downcast=True, on_unused_input='ignore'
)

rnn_test_y_evaluate = theano.function(
        inputs= [x_seq],
        outputs = y_seq
        #allow_input_downcast=True, on_unused_input='ignore'
)

rnn_test_parm = theano.function(
        inputs= [],
        outputs = [Wi,bh,Wo,bo,Wh]
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


def load_parm(PARM):
        parameters = PARM

######################################################## testbench part ##########################################
f = open('DNN.txt','r')
#f = open('posteriorgram/train.post','r')
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
wav=[]
for i in range( len(name) ):
    if i != len(name)-1:
        if int(name[i][2])+1 != int(name[i+1][2]):
            wav.append(i)
    else:
        wav.append(i)

anstype = ["aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d", "dh", "dx", "eh", "el"
               , "en", "epi", "er", "ey", "f", "g", "hh", "ih", "ix", "iy", "jh", "k", "l", "m", "ng"
               , "n", "ow", "oy", "p", "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w"
               , "y", "zh", "z"]

train_number = 1091215 #1124823
validation_num = 1124823-1091215

c = MAP()
ACC = 0
epoch=0
mask = []
i=0
try:
    while True:
        X = []
        Y = []
        count777 = 0
        flag_wav_end = 0
        flag_data_end = 0
        wav_len = 0
        if i>=train_number:#i>=1124823:
            i=0

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
                    epoch=1
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
            if i % 100000 == 0:
                print i

        if (i not in wav):
            print "FUCKING!!!!"
            print 'i' ,i
        if (len(X)!=777 or len(Y)!=777):
            print 'len wrong!!'

        i=i+1

        mask = np.ones(wav_len).tolist()+np.zeros(777-wav_len).tolist()
        rnn_train(X,Y,mask)

        if epoch==1:
            print "COST=", rnn_test_cost(X,Y,mask)
            print rnn_test_parm()
            err=0.0
            m=0
            test_index=0
            first = 1
            while(m<validation_num):
                X_test=[]
                Y_test=[]
                flag_data_end_test = 0
                flag_wav_end_test = 0
                count777_test = 0
                wave_lengh = 0
                mask_a =[]
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
                                X_test.append(f_DNNsoft[train_number+m][1:49])
                                flag_wav_end_test = 1
                                wave_lengh = int(name[train_number+m][2])
                            else:
                                y=[0]*48
                                Y_test.append(y)
                                X_test.append(y)
                    count777_test = count777_test+1;
                if ((m+train_number) not in wav):
                    print "FUCKING TWO!!!!"
                    print 'm' ,m
                if (len(X_test)!=777 or len(Y_test)!=777):
                    print 'len wrong!!'

                m=m+1
                Ya = rnn_test_y_evaluate(X)
                #mask_a = np.ones(wave_lengh).tolist()+np.zeros(777-wave_lengh).tolist()
                #haha  = rnn_test_y_modify(X,mask_a)
                #if first==1:
                    #print 'Ya',Ya
                    #print 'haha',haha
                    #print 'wave_lengh',wave_lengh
                    #print 'Ya[0]',Ya[0]
                    #first = 0
                #print "wave_lengh",wave_lengh
                for index in range(wave_lengh):
                    #if index == wave_lengh-1:
                    #        print 'input',f_DNNsoft[train_number+test_index+index][0]
                    #        print 'ans',str(ans[train_number+test_index+index].split('\n')[0])
                    if( c.map(Ya[index]) !=  str(ans[train_number+test_index+index].split('\n')[0]) ):
                        err = err+1
                        #print Ya[index]
                        #print train_number+m-wave_lengh+index
                        #if(test_index<1000):
                        #    print "test_name",name[train_number+test_index+index][0],name[train_number+test_index+index][1]
                        #    print str(ans[train_number+test_index+index].split('\n')[0])
                        #    print "y_evaluate",c.map(Ya[index])
                #print m
                test_index = test_index+wave_lengh
            print 'err',err
            ACC = 1.0-err/validation_num
            print 'ACC = %f'%(ACC)
            epoch = 0
            counter = 0
except KeyboardInterrupt:
    pass


f.close()
ans_data.close()

f_P = file('parameter_RNN_1103.txt', 'wb')
cPickle.dump(parameters, f_P, protocol=cPickle.HIGHEST_PROTOCOL)
f_P.close()
'''
######################## test ############################
parm_data = file('parameter_W_RNN.txt','rb')
parm = cPickle.load(parm_data)
load_parm(parm)

test_ans = open('RNN_test_ans_1102.csv','w')


test_c = MAP()
Y=None
m=0
test_index=0
test_ans.write('Id,Prediction\n')
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
                    X_test.append(f_DNNsoft[train_number+m][1:49])
                    flag_wav_end_test = 1
                    wave_lengh = int(name[train_number+m][2])
                else:
                    y=[0]*48
                    Y_test.append(y)
                    X_test.append(y)
        count777_test = count777_test+1;
    m=m+1
    Ya = rnn_test_y_evaluate(X)
    for index in range(wave_lengh):
        test_ans.write(f_DNNsoft[train_number+test_index+index][0])
        test_ans.write(',')
        test_ans.write(test_c.map(Ya[index],1).split('\n')[0])
        if m!=validation_num-1-1:
            test_ans.write('\n')
    test_index = test_index+wave_lengh

'''
