from sklearn import tree
 
import pandas
import numpy
import math
import xgboost
from sklearn.metrics import accuracy_score
    
    
dataframe_train = pandas.read_csv("training.csv", header=None)
dataset_train = dataframe_train.values
 
dataframe_test = pandas.read_csv("test.csv", header=None)
dataset_test = dataframe_test.values


for i in range(250001):
    if dataset_train [i,32]=='s':
          dataset_train [i,32]='1'
    else:
         dataset_train [i,32]='0'
         
    
print dataset_train.shape
print dataset_test.shape
X_test = dataset_train[1:10001,:31].astype(float)
Y_test = dataset_train[1:10001,32].astype(float)

X_train = dataset_train[10001:200001,:31].astype(float)
Y_train = dataset_train[10001:200001,32].astype(float)

W = dataset_train[1:10001,31].astype(float)     

model = xgboost.XGBClassifier()
model.fit(X_train, Y_train)


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#Accuracy: 83.00%

cf=0.
sf=0
bf=0
for i in range(len(X_test)):
    if y_pred[i]== Y_test[i]:
        cf+=1
    if y_pred[i]==1:
        if Y_test[i]==1:
            sf= sf + W[i]
        if Y_test[i]==0:
            bf= bf + W[i]
score_F=cf*100/len(X_test) 
radicandF = 2 *( (sf+bf+10) * math.log (1.0 + sf/(bf+10)) -sf)
AMS_F = math.sqrt(radicandF)
