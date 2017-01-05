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
X_test = dataset_train[1:10001,:31].astype(float)
Y_test = dataset_train[1:10001,32].astype(float)

X_train = dataset_train[10001:200001,:31].astype(float)
Y_train = dataset_train[10001:200001,32].astype(float)

W = dataset_train[1:10000,31].astype(float)


reg = tree.DecisionTreeRegressor ()
reg = reg.fit(X_train,Y_train)
R = reg.predict(X_test)
reg.score(X_test,Y_test)
cr=0.
sr=0
br=0
for i in range(len(X_test)):
    if R[i]== Y_test[i]:
        cr+=1
    if R[i]==1:
        if Y_test[i]==1:
            sr= sr + W[i]
        if Y_test[i]==0:
            br= br + W[i]
score_R=cr*100/len(X_test) 
radicandR = 2 *( (sr+br+10) * math.log (1.0 + sr/(br+10)) -sr)
AMS_R = math.sqrt(radicandR)
print score_R
print 'AMS_R =',AMS_R
#71.07
#AMS_R = 0.273223317766

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(base_estimator=reg,
                         n_estimators=50, 
                         learning_rate=0.1, random_state=0)
ada = ada.fit(X_train, Y_train)
A=ada.predict(X_test)
ca=0.
sa=0
ba=0
for i in range(len(X_test)):
    if A[i]== Y_test[i]:
        ca+=1
    if A[i]==1:
        if Y_test[i]==1:
            sa= sa + W[i]
        if Y_test[i]==0:
            ba= ba + W[i]
score_A=ca*100/len(X_test) 
radicandA = 2 *( (sa+ba+10) * math.log (1.0 + sa/(ba+10)) -sa)
AMS_A = math.sqrt(radicandA)
print score_A
print 'AMS_A =',AMS_A
#83 with 100 estimators
#82.5 with 50 estimators, AMS_A=0.505636131249
#79 with 10 estimators

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
#70.65
