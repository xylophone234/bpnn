import os
from PIL import Image
import numpy as np

path='faces/faces_4/an2i'
trainx=[]
trainy=[]
for filename in os.listdir(path):
	pixel=[]
	im=Image.open(path+'/'+filename)
	for i in range(im.size[0]):
		row=[]
		for j in range(im.size[1]):
			row.append(im.getpixel((i,j)))
		pixel.append(row)
	trainx.append(pixel)
	director=filename.split('_')[1]
	if director=='left':
		trainy.append([1,0,0,0])
	elif director=='right':
		trainy.append([0,1,0,0])
	elif director=='straight':
		trainy.append([0,0,1,0])
	elif director=='up':
		trainy.append([0,0,0,1])
trainx=np.array(trainx)	
trainy=np.array(trainy)

trainx=np.transpose(trainx.reshape((-1,32*30))-128)/256.0
trainy=np.transpose(trainy)