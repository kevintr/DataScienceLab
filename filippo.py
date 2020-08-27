


################################
#Ho ripreso la preparazione fatta da kevin e fatta la classif OVR

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import NearMiss
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

training =pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")    
# verifica valori null all'interno del training
training = training.dropna()

training_test=training.groupby(['VAR_CLASS']).size().reset_index(name='counts')
training_test.head()
training_test.plot(kind='bar',x='VAR_CLASS',y='counts')
training_test.plot(x='VAR_CLASS',y='counts')
training_test.plot(kind='hist',x='VAR_CLASS',y='counts')
training_test.plot(kind='kde',x='VAR_CLASS',y='counts')




len(training['KIT_ID'].unique())
counter = Counter(training['KIT_ID'])
print(counter)

#Trasformazione TS in datetime
training['TS'] = pd.to_datetime(training['TS'])

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


def prepareTraining():
    training = pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")    
    #da inserire il TS
    X = training[['USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI']]
    y = training['VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)

X,y = prepareTraining()
############## sono ripartito da qua#######################################
#https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/



#######prova 1
undersample = NearMiss(version=1)
#oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = undersample.fit_resample(X, y)
counter = Counter(y)
print(counter) #classi bilanciate(?)
##################
#################prova 2
#nm = NearMiss()
#X_res, y_res = nm.fit_resample(X, y)
#print('Resampled dataset shape %s' % Counter(y_res))
#lentisssimo 

###################
training_test2=training.groupby(['VAR_CLASS']).size().reset_index(name='counts')
training_test.head()
training_test.plot(kind='bar',x='VAR_CLASS',y='counts')
training_test.plot(x='VAR_CLASS',y='counts')
training_test.plot(kind='hist',x='VAR_CLASS',y='counts')
training_test.plot(kind='kde',x='VAR_CLASS',y='counts')

# logistic regression for multi-class classification using a one-vs-rest
# define dataset

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=100)
# define model
model = LogisticRegression()
# define the ovr strategy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(X, y)
# make predictions
yhat = ovr.predict(X)
yhat
y_pred = ovr.predict(X)
accuracy_score(y, y_pred)#0.696 con random state=1 con radom state=100 -> 0.652

#########
# logistic regression for multi-class classification using built-in one-vs-rest
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression(multi_class='ovr')
# fit model
model.fit(X, y)
# make predictions
yhat2 = model.predict(X)
yhat2
y_pred = model.predict(X)
accuracy_score(y, y_pred) #0.696

recall_score(y, y_pred)
###################################
####SVC - C-Support Vector Classification-
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = OneVsRestClassifier(SVC())
# fit model
model.fit(X, y)
# make predictions
yhat2 = model.predict(X)
yhat2
y_pred = model.predict(X)
accuracy_score(y, y_pred) #0.893
####################SVC LINEAR
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = OneVsRestClassifier(LinearSVC())
# fit model
model.fit(X, y)
# make predictions
yhat2 = model.predict(X)
yhat2
y_pred = model.predict(X)
accuracy_score(y, y_pred) #0.705


#########################
confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
y_true


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred)) 









