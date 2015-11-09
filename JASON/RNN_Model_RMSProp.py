import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
import cPickle
from  output48_39 import *
import pdb
__author__= 'JasonWu'

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 128
# input
N_INPUT = 48
# output
N_OUTPUT = 48

x_seq = T.matrix()
y_hat = T.matrix()
mask = T.vector()
start = T.scalar()
PARM = T.matrix()

#################### LOAD PARAMETER #################
parm_data = file('parameter_RNN_1107.txt','rb')
parm = cPickle.load(parm_data)
Wi = parm[0] 
bh = parm[1] 
Wo = parm[2] 
bo = parm[3] 
Wh = parm[4] 

'''
Wi = theano.shared( np.random.randn(N_INPUT,N_HIDDEN)*100 )
bh = theano.shared( np.zeros(N_HIDDEN) )
Wo = theano.shared( np.random.randn(N_HIDDEN,N_OUTPUT)*100 )
bo = theano.shared( np.zeros(N_OUTPUT) )
Wh = theano.shared( np.zeros( (N_HIDDEN,N_HIDDEN) ) )
'''
#sigma = theano.shared(np.random.randn(N_INPUT,N_HIDDEN) )
#Wi = theano.shared( np.random.normal(0, 0.1, (N_INPUT,N_HIDDEN)) )
#Wo = theano.shared( np.random.normal(0, 0.1, (N_HIDDEN,N_OUTPUT)) )
parameters = [Wi,bh,Wo,bo,Wh]
#sigma = [Wi,bh,Wo,bo,Wh]

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
parm_temp = PARM
#cost = -1*((T.log(y_seq_modify)*y_hat).sum())
#decay_rate = 0.999

#gradients = T.grad(cost,parameters)


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
        updates.append((p, p - lr * g))
    return updates

'''
def MyUpdate(parameters,gradients,sigma):
    mu =  np.float32(0.00001)
    parameters_updates = [(p,p - mu/s * T.clip(g,10,10)) for p,g,s in izip(parameters,gradients,sigma) ] 
    return parameters_updates

def MyUpdate_sigma(sigma,gradients):
    sigma_updates = [(p, T.sqrt(decay_rate*(p**2)+(1-decay_rate)*(g**2)) ) for p,g in izip(sigma,gradients) ] 
    return sigma_updates

def MyUpdate_sigma_initial(sigma,gradients):
    sigma_updates_inital = [(p, g) for p,g in izip(sigma,gradients) ] 
    return sigma_updates_inital
'''
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

#rnn_load_parm_2(parm)
#print rnn_test_parm()
'''
f = open('DNN_softmax.txt','r')
#f = open('train_wav1.ark','r')
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

len_max = 777

c = MAP()
ACC = 0.0
epoch=0
mask = []
i=0
epoch_counter = 47
start=1
try:
    print 'aaa'
    while True:
        X = []
        Y = []
        count777 = 0
        flag_wav_end = 0
        flag_data_end = 0
        wav_len = 0
        if i>=train_number:#i>=1124823:
            i=0
        
        while (count777<len_max) :
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
        if (len(X)!=len_max or len(Y)!=len_max):
            print 'len wrong!!'
            
        i=i+1
        #print '##############################wav_len' , wav_len
        #print '##############################i' , i
        mask = np.ones(wav_len).tolist()+np.zeros(len_max-wav_len).tolist()
        
        rnn_train_test(X,Y,mask)
        #print "COST=", rnn_test_cost(X,Y,mask)
        #Ya = rnn_test_y_evaluate(X)
        
        if epoch==1:
            epoch_counter = epoch_counter+1
            print 'epoch_counter',epoch_counter
            print "COST=", rnn_test_cost(X,Y,mask)
            #print rnn_test_parm()
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
                Ya = rnn_test_y_evaluate(X_test)
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

f_P = file('parameter_RNN_1107.txt', 'wb')
cPickle.dump(parameters, f_P, protocol=cPickle.HIGHEST_PROTOCOL)
f_P.close()
'''
######################## test ############################
f = open('DNN_test_softmax.txt','r')
test_ans = open('RNN_test_ans_1107_1.csv','w')

f_test = []
name = []
for line in f:
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
                #typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                #y=[0]*48
                #y[typeidx]=1
                #Y_test.append(y)
                X_test.append(f_test[m][1:49])
                flag_data_end_test = 1
                wave_lengh = int(name[m][2])
            else:
                y=[0]*48
                #Y_test.append(y)
                X_test.append(y)
        else:
            if(name[m][0]==name[m+1][0] and name[m][1]==name[m+1][1]) :
                #typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                #y=[0]*48
                #y[typeidx]=1
                #Y_test.append(y)
                X_test.append(f_test[m][1:49])
                m=m+1
            else: 
                if(flag_wav_end_test==0):
                    #typeidx = anstype.index(str(ans[train_number+m].split('\n')[0]))
                    #y=[0]*48
                    #y[typeidx]=1
                    #Y_test.append(y)
                    X_test.append(f_test[m][1:49])
                    flag_wav_end_test = 1
                    wave_lengh = int(name[m][2])
                else:
                    y=[0]*48
                    #Y_test.append(y)
                    X_test.append(y)
        count777_test = count777_test+1;
    m=m+1
    Ya = rnn_test_y_evaluate(X_test)
    for index in range(wave_lengh):
        test_ans.write(f_test[test_index+index][0])
        test_ans.write(',')
        test_ans.write(test_c.map(Ya[index],0).split('\n')[0])
        if m!=test_num-1-1:
            test_ans.write('\n')
    test_index = test_index+wave_lengh


