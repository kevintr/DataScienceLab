


################################

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn as sk
import datetime
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
logistic_regression=accuracy_score(y, y_pred)#0.696 con random state=1 con radom state=100 -> 0.652
logistic_regression
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
logistic_regression_2= accuracy_score(y, y_pred) #0.696
logistic_regression_2

#recall_score(y, y_pred)
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
SVC=accuracy_score(y, y_pred) #0.893
SVC
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
SVC_LINEAR=accuracy_score(y, y_pred) #0.705
SVC_LINEAR
###########################

#RandomForestClassifier

#define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=100)
# fit model
model.fit(X, y)
# make predictions
yhat2 = model.predict(X)
yhat2
y_pred = model.predict(X)
RandomForest= accuracy_score(y, y_pred) #0.736
RandomForest

############
#neurale 

#define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X, y)
NN.predict(X[460:,:])
round(NN.score(X,y), 4)#0.816


#########################conversione data secondi
df_time = pd.to_datetime(training['TS'])

second= (df_time.dt.hour*60+df_time.dt.minute)*60 + df_time.dt.second+df_time.dt.day*86400

second.head()
second.tail()
##################################################



training['VAR_CLASS'].value_counts()
prova= training[training['VAR_CLASS']==2]
prova
LOL=training.groupby('KIT_ID')
LOL

training = training().reset_index()
grouped = training.groupby('KIT_ID')['VAR_CLASS']
grouped
#loans.groupby('country_code')['loan_amount'].
#df.groupby(['Animal']).mean()
#de = sf[sf["lang"] == "de"]

test=pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\test.csv",";")
test

test_1 = test.groupby('KIT_ID')
test_1


