from sklearn import tree

import pandas
import numpy

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

X_train = dataset_train[1:100000,:31].astype(float)
Y_train = dataset_train[1:100000,32].astype(float)

X_test = dataset_train[100001:250001,:31].astype(float)
Y_test = dataset_train[100001:250001,32].astype(float)


reg = tree.DecisionTreeRegressor ()
reg = reg.fit(X_train,Y_train)
R = reg.predict(X_test)
reg.score(X_test,Y_test)
cr=0.
for i in range(len(X_test)):
    if R[i]== Y_test[i]:
        cr+=1
score_R=cr*100/len(X_test)   
print score_R
#72.686

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(base_estimator=reg,
                         n_estimators=50, 
                         learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, Y_train)
A=ada.predict(X_test)
ca=0
for i in range(len(X_test)):
    if A[i]== Y_test[i]:
        ca+=1
score_A=ca*100/len(X_test)
print score_A
#82

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 50,random_state=0)
forest = forest.fit(X_train,Y_train)
F=forest.predict(X_test)
cf=0
for i in range(len(X_test)):
    if F[i]== Y_test[i]:
        cf+=1
score_F=cf*100/len(X_test)
print score_F
                         
clf = tree.DecisionTreeClassifier ()
clf = clf.fit(X_train,Y_train)

C = clf.predict(X_test)
cc=0.
for i in range(len(X_test)):
    if C[i]== Y_test[i]:
        cc+=1
score_C=cc*100/len(X_test)
print score_C
#73.5473333333
