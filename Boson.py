# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold

import numpy
import pandas
from keras.regularizers import l1, activity_l1
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from keras.utils import np_utils

# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[1:10000:,1:32].astype(float)
Y = dataset[1:10000:,32].astype(float)
X2 = dataset[10000:50000:,1:32].astype(float)
Y2 = dataset[10000:50000:,32].astype(float)

Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
#Y2 = np_utils.to_categorical(Y2, 2)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X2, Y2):
	# create model
	
	Y3 = np_utils.to_categorical(Y2, 2) # convert class vectors to binary class matrices
	
	model = Sequential()
	model.add(Dense(30, input_dim=31, init='normal', activation='relu' ,W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
	model.add(Dense(30, activation='relu' ))
	model.add(Dense(30, activation ='relu'))
	model.add(Dense(2))
	model.add(Activation('softmax'))
	#model.add(Activation('sigmoid'))

	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#sgd = SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
	rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.000)
	model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
	#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
	#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent



	# Fit the model

	model.fit(X2[train], Y3[train], nb_epoch=100, batch_size=96)

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
print(Z[1:10])
print(Y[1:10])

#----------------------------------------------------------------------------
# TO DO
# CV bagging DONE
# Momentum
# Learning rate
# L1 penalty DONE
# L2 penalty
# AMS metric NECESSAIRE?
# Advanced features
# Winner takes all activation 
# Constrain neurons in first layer
#
#
#
#----------------------------------------------------------------------------
