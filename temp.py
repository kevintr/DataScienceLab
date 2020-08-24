# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os #os è un libreria per induviduare la directory dove ci si trova
from datetime import datetime
from sklearn import tree
#import weka.core.jvm as jvm
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import MultinomialNB
import sklearn
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
import numpy
from imblearn.under_sampling import NearMiss

# Training set upload
training = pd.read_csv('training.csv', sep=';')    
# verifica valori null all'interno del training
training.dropna()

#Trasformazione TS in datetime
test['TS'] = pd.to_datetime(test['TS'])

#training['TS'] = pd.to_numeric(training['TS'], downcast='float', errors='ignore')

# type(training.loc[0,'TS'] )
training.loc[0,'TS']

print('0 ' + str(len(training[training['VAR_CLASS'] == 0])))#16521526
print('1 ' + str(len(training[training['VAR_CLASS'] == 1])))#36
print('2 ' + str(len(training[training['VAR_CLASS'] == 2])))#472

str(len(training[training['VAR_CLASS'] == 2]))

#Analisi TS
training['TS'] = pd.to_datetime(training['TS'])
training['TS'].dt.year.unique()#2018
training['TS'].dt.month.unique()#11
training['TS'].dt.day.unique()#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
training['VAR_CLASS'].unique()#1,2,3
training[training['NUM_CLI'] >= 100]


# training['USAGE'].sort_values(ascending=True)

trainingOrderByUSAGE = training.sort_values(by=['USAGE'])
trainingOrderByAVG = training.sort_values(by=['AVG_SPEED_DW'])
training.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))
training['KIT_ID'].describe()

descriptiveQuantity = training[['USAGE','AVG_SPEED_DW','NUM_CLI']].describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

descriptiveQuantity

#da inserire il TS
training = training[['USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI','VAR_CLASS']]
training


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#In terms of machine learning, Clf is an estimator instance, which is used to store model.
#We use clf to store trained model values, which are further used to predict value, based on the previously stored weights.


# Generate a synthetic imbalanced classification dataset
#Synthetic Minority Over-sampling Technique
oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)

#split del dataset in trainset e test_set
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

modelLR = LogisticRegression()
modelLR.fit(x_train, y_train)
y_pred = modelLR.predict(x_test)
accuracy_score(y_test, y_pred)#0.43

#recall_score(y_test, y_pred)

y_true = y

confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
y_true


#ovr
#########################OneVsRestClassifier############################################
#al posto di clf possiamo mettere qualsiasi altra roba
clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred) #0.2


##############################OVO#########################################################
model = SVC(decision_function_shape='ovo')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)



# define model
model = SVC()
# define ovo strategy
ovo = OneVsOneClassifier(model)
# fit model
ovo.fit(X, y)
# make predictions
y_pred = ovo.predict(X)



#########################################################################################################
#Unire 1 a 2 e formare un unico pezzo e provare l'algoritmo binario######################################
#########################################################################################################
training['VAR_CLASS'] = training['VAR_CLASS'].replace(2,1)
#Adesso il problema diventa binario ed è così possibile usare gli algoritmi più noti
training['VAR_CLASS']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

nr = NearMiss()
X_train, y_train = nr.fit_sample(x_train, y_train)
counter = Counter(y_train)
print(counter)

model = SVC(decision_function_shape='ovo')
# fit model
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(x_test)

confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
accuracy_score(y_test, y_pred)

clf = OneVsRestClassifier(SVC())
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred)


clf = MultinomialNB(alpha=1)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred)

#This approach is commonly used for algorithms that naturally predict numerical class 
#membership probability or score, such as: 
#Perceptron
#As such, the implementation of these algorithms in the scikit-learn library implements the OvR strategy 
#by default when using these algorithms for multi-class classification.























########################################Parte relatica alla lettura del TEST.csv
#test['TS'] = pd.to_numeric(test['TS'], downcast='float', errors='ignore')
#test = pd.read_csv('test.csv', sep=';')       