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
from sklearn.svm import SVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import imblearn
from imblearn.over_sampling import SMOTE

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


e = training[training['VAR_CLASS'] == 1]

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

np.unique(y_pred)




training = pd.read_csv('training.csv', sep=';') 
training[training['VAR_CLASS'] ==2 ]
training1 = training.iloc[3238758:3239770,:]
training1 
X= training1.iloc[:, 1:4].values
Y = training1['VAR_CLASS']
Y

OneVsRestClassifier(LinearSVC(random_state=0,interazione)).fit(X, y).predict(X)



X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo',gamma='auto')
clf.fit(X, Y)
SVC(decision_function_shape='ovo')
dec = clf.decision_function([[3]])
dec.shape[1] # 4 classes: 4*3/2 = 6
6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes
4


from sklearn import svm


clf = svm.SVC()
clf.fit(X, y)
SVC()


# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)


clf.predict([[69, 10]])


X_train, X_test, y_train, y_test = train_test_split(training[(['USAGE','AVG_SPEED_DW','NUM_CLI'])],
                                                    training['VAR_CLASS'],
                                                    test_size=0.33,
                                                    random_state=42)
# multiclass con python
# binarizzazione con python
# preprocessing-- dataset sbilanciati 
#    matrici di costo
#    ribilanciamento- oversampling
# split training.csv splittare il dataset 70% 30%

  
X = training1[(['USAGE','AVG_SPEED_DW','NUM_CLI'])]
y = training1['VAR_CLASS']


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [0])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y = onehotencoder.fit_transform(y).toarray()





from sklearn.linear_model import LogisticRegression


training1 = training.iloc[3238758:3239770,:]
print('0 ' + str(len(training1[training1['VAR_CLASS'] == 0])))
print('1 ' + str(len(training1[training1['VAR_CLASS'] == 1])))
print('2 ' + str(len(training1[training1['VAR_CLASS'] == 2])))


X = training[(['USAGE','AVG_SPEED_DW','NUM_CLI'])]
y = training['VAR_CLASS']

clf = LogisticRegression(random_state=0).fit(X, y)

np.unique(clf.predict(X))

array([0, 0])
 clf.predict_proba(X[:2, :])
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
       [9.7...e-01, 2.8...e-02, ...e-08]])
 clf.score(X, y)
0.97...



# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
import numpy

# define dataset
X1, y1 = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)


X = X.to_numpy()
y = y.array  
y = y.to_numpy()

# summarize class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()



oversample = SMOTE(random_state=42,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)


lr = LogisticRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
accuracy_score(y, y_pred)

y_true = y

confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

y_true
