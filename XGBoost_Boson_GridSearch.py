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
X_test = dataset_test[1:10001,:31].astype(float)
Y_test = dataset_train[1:10001,32].astype(float)

X_train = dataset_train[10001:250001,:31].astype(float)
Y_train = dataset_train[10001:250001,32].astype(float)

W = dataset_train[10001:250001,31].astype(float)    

def AMS(estimator,X,y):
    G=estimator.predict(X)

    cg=0.
    sg=0
    bg=0
    for i in range(len(y)):
        if G[i]== y[i]:
            cg+=1
        if G[i]==1:
            if y[i]==1:
                sg= sg + W[i]
            if y[i]==0:
                bg= bg + W[i]
    score_G=cg*100/len(y) 
    radicandG = 2 *( (sg+bg+10) * math.log (1.0 + sg/(bg+10)) -sg)
    AMS_G = math.sqrt(radicandG)
    print ('Pourcentage = ',score_G,'%')
    print ('AMS =',AMS_G)
    return AMS_G

parameters = {'max_depth':range(3,10,1),'n_estimators':range(25,250,25),'learning_rate':[0.05,0.075,0.1,0.125,0.15]}

gradient = xgboost.XGBClassifier()

#on fait un Gridsearch
best_grad = grid_search.GridSearchCV(gradient, parameters, scoring=AMS)
best_grad.fit(X_train, Y_train)
print("Best: %f using %s" % (best_grad.best_score_, best_grad.best_params_))
for params, mean_score, scores in best_grad.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    
