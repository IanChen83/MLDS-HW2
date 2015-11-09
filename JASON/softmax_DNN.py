import numpy as np
import theano
import theano.tensor as T

def float_convert(i):
	try: 
		return np.float32(i)
	except ValueError :
		return i

f = open('DNN_test.txt','r')
f_out = open('DNN_softmax_test.txt','w')
f_DNN = []

for line in f:
	input_x = line.split()
	input_x = [float_convert(i) for i in input_x]
	f_DNN.append(input_x)

x = T.vector()
y = T.exp(x)
exp = theano.function([x],y)


for i in range(len(f_DNN)):
	x = exp(f_DNN[i][1:70])
	x = x/x.sum()
	f_out.write(f_DNN[i][0])
	f_out.write(' ')
	for i in range(len(x)):
		f_out.write(str(x[i]))
		f_out.write(' ')
	if i!=len(f_DNN)-1:
		f_out.write('\n')

f_out.close()
f.close()