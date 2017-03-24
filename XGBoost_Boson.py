from sklearn import tree
 
import pandas
import numpy
import math
import xgboost
from sklearn.metrics import accuracy_score
from sklearn import grid_search
    
    
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
X_test = dataset_test[1:20001,:31].astype(float)
Y_test = dataset_train[1:20001,32].astype(float)

X_train = dataset_train[20001:250001,:31].astype(float)
Y_train = dataset_train[20001:250001,32].astype(float)

W = dataset_train[1:20001,31].astype(float)


model = xgboost.XGBClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

T=model.predict_proba(X_test)[:,1]
print (T)
print (T.shape)
print (Y_test[1])
temp = T.argsort()
ranks = numpy.empty(len(T), int)
ranks[temp] = numpy.arange(len(T)) +1

S=[]
for i in range(len(T)):
    if T[i]>0.5:
        S.append('s')
    else:
        S.append('b')
x=1
res = [["EventId","RankOrder","Class"]]
for i in range(len(T)):
    res.append([int(X_test[i,0]),ranks[i],S[i]])
    
print len(res)       
import csv
c= csv.writer(open("result.csv","wb"))

for i in res:
    c.writerow(i)

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

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

print AMS_F
