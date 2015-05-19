import numpy as np
import math

trainSet=[[0.1,0.1,-0.9],[0.9,0.9,-0.9],[0.1,0.9,0.9],[0.9,0.1,0.9]]
# trainSet=[[0.1,0.1,0.1],[0.1,0.9,0.1],[0.9,0.1,0.1],[0.9,0.9,0.9]]
trainSet=np.array(trainSet)
trainSet=np.transpose(trainSet)

def sigmoid(x):
	# return 1.0/(1.0+math.exp(-x))
	return math.tanh(x)
def dsigmoid(y):
	# return y*(1.0-y)
	return 1.0-y**2

sigmoid_ufunc=np.vectorize(sigmoid,otypes=[np.float])
dsigmoid_ufunc=np.vectorize(dsigmoid,otypes=[np.float])
# alpha=0.3

class Layer:
	def __init__(self,n_input,n_output):
		self.n_input=n_input
		self.n_output=n_output
		self.w=np.random.rand(n_output,n_input)*0.2-0.1
		self.b=(np.random.rand(n_output)).reshape(-1,1)
		self.z=None
		self.input=None
		self.output=None
		self.delta=None
		self.deltaw=None
		self.deltab=None
		self.lastDeltaw=np.zeros((n_output,n_input))
		self.lastDeltab=np.zeros((n_output,)).reshape(-1,1)


	def forward(self,x):
		self.input=x
		out=np.dot(self.w,self.input)
		self.z=out+self.b
		self.output=sigmoid_ufunc(self.z)
		# print('w=',self.w)
		# print('b=',self.b)
		# print('input=,',self.input)
		# print('output=,',self.output)
		return self.output

	def adjust(self,delta,alpha,m):
		# print('delta',delta)
		self.delta=delta
		# self.deltaw=np.dot(self.delta,np.transpose(self.output)).sum(axis=1).reshape(-1,1) 
		self.deltaw=np.dot(self.delta,np.transpose(self.input))/self.input.shape[1]
		# self.deltaw=(self.delta*self.input).sum(axis=1).reshape(-1,1)
		self.deltab=self.delta.sum(axis=1).reshape(-1,1)/self.input.shape[1]
		# print (self.deltaw)
		self.w=self.w+self.deltaw*alpha+m*self.lastDeltaw
		self.lastDeltaw=self.deltaw
		# print('b',self.b)
		# print('delta',self.deltab.sum(axis=1))
		self.b=self.b+self.deltab*alpha+m*self.lastDeltab
		self.lastDeltab=self.deltab

class BPNN:
	def __init__(self,n_input,n_hidden,n_output):
		self.hiddenLayer=Layer(n_input,n_hidden)
		self.outputLayer=Layer(n_hidden,n_output)

	def forward(self,x):
		h=self.hiddenLayer.forward(x)
		o=self.outputLayer.forward(h)
		return o

	def train(self,trainSetx,trainSety,n,alpha=0.1,m=0.3):
		for i in range(n):
			o=self.forward(trainSetx)
			# print('o',o)
			deltao=(trainSety-o)*dsigmoid_ufunc(self.outputLayer.output)
			# print('deltao',deltao)
			deltah=np.dot(np.transpose(self.outputLayer.w),deltao)*dsigmoid_ufunc(self.hiddenLayer.output)
			# print('deltah',deltah)
			self.outputLayer.adjust(deltao,alpha,m)
			self.hiddenLayer.adjust(deltah,alpha,m)
			# print(o)
			print(((trainSety-o)*(trainSety-o)).sum())

