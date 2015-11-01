import theano
import theano.tensor as T
import numpy as np
__author__= 'jason'

class MAP:
	"""docstring for ClassName"""
	def __init__(self):
		self.map_data = open('48_39.map','r')
		self.in_48 = []
		self.in_39 = []	
		for line in self.map_data:
			in_x = line.split('\t')
			self.in_48.append(in_x[0])
			self.in_39.append(in_x[1])	

	def map(self,z,index_in=0):
		y = z.tolist()
		big = max(y)
		big_index = y.index(big)
		#print big_index
		if index_in==0:
			return self.in_48[big_index]
		else :
			return self.in_39[big_index]
#m = MAP()
#print m.map([1,2,3,4])
