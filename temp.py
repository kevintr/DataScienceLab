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
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
clf.fit(X, y).predict(X)

#In terms of machine learning, Clf is an estimator instance, which is used to store model.

#We use clf to store trained model values, which are further used to predict value, based on the previously stored weights.

#path = os.getcwd().replace("\\","/") # inversione dello slash
#path
#print(path + '/test.csv')

test = pd.read_csv('test.csv', sep=';')       
training = pd.read_csv('training.csv', sep=';')    

test

# training.TS.strptime
# datetime.strptime(training.TS '%d/%m/%y %H:%M:%S')
# Trasformazione TS in datetime
#training['TS'] = pd.to_datetime(training['TS'])
#test['TS'] = pd.to_datetime(test['TS'])

training['TS'] = pd.to_numeric(training['TS'], downcast='float', errors='ignore')
test['TS'] = pd.to_numeric(test['TS'], downcast='float', errors='ignore')



# type(training.loc[0,'TS'] )
training.loc[0,'TS']

training[training['VAR_CLASS'] == 2]

print('0 ' + str(len(training[training['VAR_CLASS'] == 0])))
print('1 ' + str(len(training[training['VAR_CLASS'] == 1])))
print('2 ' + str(len(training[training['VAR_CLASS'] == 2])))

training[training['VAR_CLASS'] == 2]

str(len(training[training['VAR_CLASS'] == 2]))

training.groupby('TS')['USAGE','AVG_SPEED_DW','NUM_CLI'].sum()   

training['TS'].dt.year.unique()
training['TS'].dt.month.unique()
training['TS'].dt.day.unique()
training['VAR_CLASS'].unique()
training[training['NUM_CLI'] >= 100]
training[training['VAR_CLASS'] == 2]



# training['USAGE'].sort_values(ascending=True)

trainingOrderByUSAGE = training.sort_values(by=['USAGE'])
trainingOrderByAVG = training.sort_values(by=['AVG_SPEED_DW'])
training.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))
training['KIT_ID'].describe()

descriptiveQuantity = training[['USAGE','AVG_SPEED_DW','NUM_CLI']].describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

training.dropna()

descriptiveQuantity


#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

training = training[['USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI','VAR_CLASS']]
training


X= training.iloc[:, 0:4].values
y = training['VAR_CLASS']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


training[training['VAR_CLASS'] == 2]

test1 = test[['USAGE','AVG_SPEED_DW','NUM_CLI']]
test1



test



X = training1.iloc[:, 0:4].values
y = test1.iloc[:, 4].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(training1)
X_test = sc.transform(test1)
X_test

X_train

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)









