# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os #os è un libreria per induviduare la directory dove ci si trova
from datetime import datetime
from sklearn import tree
#import weka.core.jvm as jvm
from sklearn.metrics import classification_report, confusion_matrix

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf

#path = os.getcwd().replace("\\","/") # inversione dello slash
#path
#print(path + '/test.csv')

test = pd.read_csv('test.csv', sep=';')       
training = pd.read_csv('training.csv', sep=';')    

test

# training.TS.strptime
# datetime.strptime(training.TS '%d/%m/%y %H:%M:%S')
# Trasformazione TS in datetime
training['TS'] = pd.to_datetime(training['TS'])
test['TS'] = pd.to_datetime(test['TS'])



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



# training['USAGE'].sort_values(ascending=True)

training.sort_values(by=['USAGE'])

training.sort_values(by=['AVG_SPEED_DW'])

training.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

training['KIT_ID'].describe()

descriptiveQuantity = training[['USAGE','AVG_SPEED_DW','NUM_CLI']].describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

training.dropna()

descriptiveQuantity

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))








