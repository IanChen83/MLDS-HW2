import theano
import theano.tensor as T
import numpy as np
#test = open('DNN_softmax_test.txt','r')
raw = open('RNN_test_ans_1111.csv','r')
ans_data = open('try2.csv','w')
'''
test_d = []
for line in test:
    x = line.split(' ')
    test_d.append(x[0])
'''
label = []
label_all = []
ans = [] 
name_1=[]
name_2=[]
name_3=[]
i =0
s=0
for line in raw:
    if s ==0 :
        s=1
    else:
        label_x = line.split(',')
        name_x = label_x[0].split('_')
        label_all.append(label_x[0])
        label.append(label_x[1])
        name_1.append(name_x[0])
        name_2.append(name_x[1])
        name_3.append(name_x[2])
'''
for i in range (len(label_all)):
    if test_d[i]!=label_all[i]:
        print i
        print test_d[i]
        print label_all[i]
'''
while i<len(label)-2:
    if (name_1[i]==name_1[i+1] and name_1[i+1]==name_1[i+2] and name_2[i]==name_2[i+1] and name_2[i+1]==name_2[i+2]) :
        if label[i]!=label[i+1] and label[i]!=label[i+2] and label[i+1]!=label[i+2]:
            if np.random.normal(0,0.1,(1,1))[0][0]>0 :
                label[i+1] = label[i]
            else:
                label[i+1] = label[i+2]
        elif label[i]==label[i+2] and label[i+1]!=label[i]:
            label[i+1]=label[i]    
    i=i+1

ans_data.write('Id,Prediction\n')
for i in range(len(label)):
    ans_data.write(label_all[i])
    ans_data.write(',')
    ans_data.write(label[i].split('\n')[0])
    if i!=len(label)-1:
        ans_data.write('\n')
