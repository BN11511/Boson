# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Layer, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import math
from math import log

from keras import backend as K
from theano import tensor as T

from sklearn.model_selection import StratifiedKFold

import numpy
import pandas
from keras.regularizers import l1, activity_l1
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from keras.utils import np_utils

class WinnerTakeAll1D_GaborMellis(Layer):

    def __init__(self, spatial=1, OneOnX = 3,**kwargs):
        self.spatial = spatial
        self.OneOnX = OneOnX
        self.uses_learning_phase = True
        super(WinnerTakeAll1D_GaborMellis, self).__init__(**kwargs)

    def call(self, x, mask=None):
        R = T.reshape(x,(T.shape(x)[0],T.shape(x)[1]/self.OneOnX,self.OneOnX))
        M = K.max(R, axis=(2), keepdims=True)
        R = K.switch(K.equal(R, M), R, 0.)
        R = T.reshape(R,(T.shape(x)[0],T.shape(x)[1]))
        return R

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)
    
L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=3)

# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values

# Let's move all the azymuth angle features at the end of X (12, 16, 19, 21, 26, 29)
# 12 <-> 28 PhiCentrality
dataset[:,[12,28]]
# 16 <-> 27 Phitau
dataset[:,[16,26]]
# 19 <-> 25 Philep
dataset[:,[19,25]]
# 21 <-> 24 Phimet
dataset[:,[21,24]]
# 26 Phijetleading
# 29 Phijetsubleading


# split into input (X) and output (Y) variables
W = dataset[1:10000:,31:32]
X = dataset[1:10000:,1:29].astype(float)
Y = dataset[1:10000:,32].astype(float)
X2 = dataset[10000:230000:,1:29].astype(float)
Y2 = dataset[10000:230000:,32].astype(float)


# Implementation of advanced features in x and x2
# Notes : minv(tau,lep) (3) | 


for i in range(9999):
	# Implementation of Phi derived features
	X[i,24] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X[i,25] = min(dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X[i,26] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24])
	X[i,27] = dataset[i,25]-dataset[i,24] # Erreur ?
	# Implementation of mass based features
	#print(dataset[1, 3])
	#x[i,4] = log(1+dataset[i, 3])
	


for i in range(220000):
	# Implementation of Phi derived features
	X2[i,24] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X2[i,25] = min(dataset[i,26]-dataset[i,24],dataset[i,25]-dataset[i,24])
	X2[i,26] = min(dataset[i,26]-dataset[i,25],dataset[i,26]-dataset[i,24])
	X2[i,27] = dataset[i,25]-dataset[i,24] # Erreur ?

	# Implementation of mass based features
	#x2[i,4] = log(1)#+dataset[i, 3])




Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
#Y2 = np_utils.to_categorical(Y2, 2)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []

model = Sequential()
#model.add(Convolution1D(32, 3, border_mode='valid', input_shape=(30,1), activation='relu'))
#model.add(MaxPooling1D(pool_length=10))
#model.add(GlobalMaxPooling1D())
#model.add(Flatten())  
model.add(Dense(600, input_dim=28, init='normal', activation='relu' ,W_regularizer=l1l2(l1=0.000005, l2=0.00005))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
model.add(L)
model.add(Dense(600, activation ='relu'))
model.add(L)
model.add(Dense(600, activation ='relu'))
model.add(L)
model.add(Dense(2))
model.add(Activation('softmax'))
#model.add(Activation('sigmoid'))

# Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#sgd = SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent

for i in range(5):
	for train, test in kfold.split(X2, Y2):
		# create model
	
		Y3 = np_utils.to_categorical(Y2, 2) # convert class vectors to binary class matrices
	
	
		# Fit the model

		model.fit(X2[train], Y3[train], nb_epoch=200, batch_size=96)

		# evaluate the model
		scores = model.evaluate(X2[test], Y3[test])
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test

#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

Z = model.predict(X, batch_size=32, verbose=0)

s = 0
b = 0

for i in range(0,9999):
	if (Z[i][1]>Z[i][0]):
		if (Y[i][1]>Y[i][0]):
			s = s + W[i][0]
		if (Y[i][1]<=Y[i][0]):
			b = b + W[i][0]

br = 10.0
radicand = 2 *( (s+b+br) * math.log(1.0 + s/(b+br)) - s)
AMS = math.sqrt(radicand)


print("AMS : ")
print(AMS)



print(Z[1:10])
print(Y[1:10])

#----------------------------------------------------------------------------
# TO DO
# CV bagging with 10 repetitions DONE
# Momentum
# Learning rate
# L1 penalty DONE
# L2 penalty
# AMS metric DONE
# Advanced features 4/10 (missing features?)
# Elimination of Azymuth angles features DONE
# Winner takes all activation DONE 
# Constrain neurons in first layer
#
#
#
#----------------------------------------------------------------------------
