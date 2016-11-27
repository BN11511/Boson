# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.regularizers import l2, activity_l2
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("training.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:10000:,1:32]
Y = dataset[:10000:,32]

# create model
model = Sequential()
model.add(Dense(100, input_dim=31, init='uniform', activation='relu' ,W_regularizer=l2(0.000005), activity_regularizer=activity_l2(0.000005)))
model.add(Dense(60, input_dim=31, init='uniform', activation='relu' ,W_regularizer=l2(0.00005), activity_regularizer=activity_l2(0.00005)))
model.add(Dense(60, input_dim=31, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='softmax'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=200, batch_size=96)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test
