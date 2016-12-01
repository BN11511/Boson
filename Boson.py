# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy
import pandas
from keras.regularizers import l1, activity_l1
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[1:10000:,1:32].astype(float)
Y = dataset[1:10000:,32].astype(float)
X2 = dataset[10000:25000:,1:32].astype(float)
Y2 = dataset[10000:25000:,32].astype(float)

# create model
model = Sequential()
model.add(Dense(30, input_dim=31, init='normal', activation='relu' ,W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
model.add(Dense(30, init='normal', activation='relu' ))
model.add(Dense(30, init='normal', activation='relu'))
model.add(Dense(1, init='normal' , activation='sigmoid'))

# Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#sgd = SGD(lr=1, momentum=0.5, decay=0.97, nesterov=True)
model.compile(optimizer="RMSprop", loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
# Fit the model

model.fit(X2, Y2, nb_epoch=200, batch_size=96)

# evaluate the model
scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test

Z = model.predict(X, batch_size=32, verbose=0)
print(Z[1:10])
print(Y[1])

