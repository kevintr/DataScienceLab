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
from sklearn.svm import *
#from sklearn import MultinomialNB
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
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import datetime, time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import make_classification as make_classification

        
training = pd.read_csv('training.csv', sep=';')  
# verifica valori null all'interno del training
training = training.dropna()
training['TS'] = pd.to_datetime(training['TS'])

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


#def prepareTraining():
#    training = pd.read_csv('training.csv', sep=';')   
#    epoch = datetime.datetime.utcfromtimestamp(0)
#    training['TS'] = pd.to_datetime(training['TS'])
#    training['TS'] = training['TS'] - epoch
#    training['TS'] = training['TS'].dt.total_seconds()
#    #da inserire il TS
#    X = training[['TS','KIT_ID','USAGE','NUM_CLI']]
#    y = training['VAR_CLASS']
#    
#    X = X.to_numpy()
#    y = y.to_numpy()
#    return (X,y)


#X,y = prepareTraining()
#In terms of machine learning, Clf is an estimator instance, which is used to store model.
#We use clf to store trained model values, which are further used to predict value, based on the previously stored weights.

# Generate a synthetic imbalanced classification dataset
#Synthetic Minority Over-sampling Technique
oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modelLR = LogisticRegression()
modelLR.fit(X_train, y_train)
y_pred = modelLR.predict(X_test)
accuracy_score(y_test, y_pred)#0.43
counter = Counter(y_pred)
print(counter)

#recall_score(y_test, y_pred)

y_true = y

confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
y_true


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
#ovr
#########################OneVsRestClassifier############################################
#al posto di clf possiamo mettere qualsiasi altra roba
clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred) #0.2


##############################OVO#########################################################
##############################OVO#########################################################
##############################OVO#########################################################
class Results:

    def __init__(self, confusion_matrix,accuracy_score,clf):
        ## private varibale or property in Python
        self.__confusion_matrix = confusion_matrix
        self.__accuracy_score = accuracy_score
        self.__clf = clf
        
    def __init__(self):
        ## private varibale or property in Python
        pass
    ## getter method to get the properties using an object
    def get_confusion_matrix(self):
        return self.__confusion_matrix

    ## setter method 
    def set_confusion_matrix(self, confusion_matrix):
        self.__confusion_matrix = confusion_matrix
        
    ## getter method to get the properties using an object
    def get_accuracy_score(self):
        return self.__accuracy_score

    ## setter method
    def set_accuracy_score(self, accuracy_score):
        self.__accuracy_score = accuracy_score
   
    ## getter method to get the properties using an object
    def get_clf(self):
        return self.clf

    ## setter method
    def set_clf(self, clf):
        self.clf = clf

#restituisce in outPut l'accuracy e la confusionMatrix per il classifier in input
def ovoClassifier(classifier):
    ovoClassifierResults = Results()
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
#   confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    ovoClassifierResults.set_accuracy_score(score)
    ovoClassifierResults.set_confusion_matrix(confusionMatrix)
    return ovoClassifierResults

X,y = prepareTraining()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train

X_train = np.delete(X_train, slice(1, 10000000), 0)
y_train = np.delete(y_train, slice(1, 10000000), 0)
len(X_train)
len(y_train)
resultLinearSVR = ovoClassifier(LinearSVR())
resultLinearSVR.get_accuracy_score()
resultLinearSVR.get_confusion_matrix()

resultLinearSVC = ovoClassifier(LinearSVC())
resultNuSVR = ovoClassifier(NuSVC())
resultLinearSVR = ovoClassifier(NuSVR())
resultOneClassSVM = ovoClassifier(OneClassSVM())
resultSVC = ovoClassifier(SVC())
resultSVR = ovoClassifier(SVR())
resultSVR = ovoClassifier(LogisticRegression()())

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

##########################################
model = SVC(decision_function_shape='ovo')
# fit model
model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train)
model
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
training['KIT_ID'][training['VAR_CLASS'] != 2]
training.where(training['KIT_ID'] ==3409364152 or training['KIT_ID']==1629361016 or training['KIT_ID']==2487219358)

training794615332 = training[training['KIT_ID'] == 794615332 ]
training794615332.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
#kit3409364152.plot(x='TS',y='AVG_SPEED_DW',color='red')#costante
training794615332.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante





####################################    PROVA PREDIZIONE CON SOLO KIT_DI CON 1E 2 #######################
def prepareTraining2(dateframe):
    epoch = datetime.datetime.utcfromtimestamp(0)
    dateframe.loc[:,'TS'] = pd.to_datetime(dateframe.loc[:,'TS'])
    dateframe.loc[:,'TS'] = dateframe.loc[:,'TS'] - epoch
    dateframe.loc[:,'TS'] = dateframe.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = dateframe.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y =  dateframe.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)



#restituisce in outPut l'accuracy e la confusionMatrix per il classifier in input
def binaryHoldOutClassifierSmote(classifier,dataframe):
    X,y = prepareTraining2(dataframe)
    #Synthetic Minority Over-sampling Technique
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
    results = Results()
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
#    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf)
    return results

def binaryCrossValidationClassifierSmote(classifier,dataframe):
    X,y = prepareTraining2(dataframe)
    #Synthetic Minority Over-sampling Technique
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    results = Results()
    clf = classifier
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X, y, cv=20)
    score = accuracy_score(y, y_pred)
    confusionMatrix = confusion_matrix(y, y_pred, labels=[0, 1])
#    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf)
    return results

def testClassifier(clf1,dataframe):
    X,y = prepareTraining2(dataframe)
    results = Results()
    y_pred = clf1.predict(X)
    score = accuracy_score(y, y_pred)
    confusionMatrix = confusion_matrix(y, y_pred, labels=[0, 1])
#   confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(score)
    print(confusionMatrix)
    results.set_accuracy_score(score)
    results.set_confusion_matrix(confusionMatrix)
    results.set_clf(clf1)
    return results

def prepareTest(test):
    epoch = datetime.datetime.utcfromtimestamp(0)
    test.loc[:,'TS'] = pd.to_datetime(test.loc[:,'TS'])
    test.loc[:,'TS'] = test.loc[:,'TS'] - epoch
    test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()
    test = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    
    #test = test_X.to_numpy()
    test = test.dropna()
    return (test)

#training e test
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
kitWith1or2.loc[:,'VAR_CLASS'] = kitWith1or2.loc[:,'VAR_CLASS'].replace(2,1)

###### RandomForestClassifier con HouldOut
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
resultRandomForest = binaryHoldOutClassifierSmote(RandomForestClassifier(n_estimators=100),kitWith1or2)

###### DecisionTreeClassifier con HouldOut
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
resultDecisionTree = binaryHoldOutClassifierSmote(DecisionTreeClassifier(),kitWith1or2)


#####  RandomForestClassifier() Cross Validation ########
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
resultRandomForest = binaryCrossValidationClassifierSmote(RandomForestClassifier(n_estimators=100),kitWith1or2)

##### DecisionTreeClassifier()  Cross Validation ########
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
resultDecisionTree = binaryCrossValidationClassifierSmote(DecisionTreeClassifier(),kitWith1or2)


# teeeeeeeeeeeeeeeeest
#kitNot1or2[kitNot1or2['VAR_CLASS']==1]

kitNot1or2 = training.loc[((training['resultTestRandomForest = binaryClassifierSmote(resultRandomForest.get_clf(),kitNot1or2)KIT_ID'] != 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]


kitNot1or2 = training.loc[((training['KIT_ID'] != 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]
resultTestDecisionTree = testClassifier(resultRandomForest.get_clf(),kitNot1or2)

kitNot1or2 = training.loc[((training['KIT_ID'] != 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]
resultTestDecisionTree = testClassifier(resultDecisionTree.get_clf(),kitNot1or2)


trainingReplace = training.replace(2,1)
trainingReplace['VAR_CLASS'].unique()
X,y = prepareTraining2(training)
print(Counter(y))
    ####################### Make classification ############################################################################
Z,w = prepareTraining2(training)
training

counter = Counter(w)
print(counter)
type()


training.loc[:,'VAR_CLASS'] = training.loc[:,'VAR_CLASS'] -1
trainingCampionato = training.sample(n=1600 , replace=False,random_state=1000) #['VAR_CLASS']
print(Counter(z['VAR_CLASS']))

df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                    'num_wings': [2, 0, 0, 0],
                    'num_specimen_seen': [10, 2, 1, 8]},
                   index=['falcon', 'dog', 'spider', 'fish'])
df


df['num_legs'].sample(n=3, random_state=1)
df.sample(frac=0.5, replace=True, random_state=1)
df.sample(n=4, weights='num_specimen_seen', random_state=1)
#############    Random Forest ##################
##Create a Gaussian Classifier
#clf=RandomForestClassifier(n_estimators=100)
#
##Train the model using the training sets y_pred=clf.predict(X_test)
#clf.fit(X_train,y_train)
#
#y_pred=clf.predict(X_test)
#confusion_matrix(y_test, y_pred, labels=[0, 1])
#accuracy_score(y_test, y_pred)
#############    Random Forest ##################
#
#############            DecisionTreeClassifier                          ##################OTTIMI RISULATI
#clf = DecisionTreeClassifier()
#
## Train Decision Tree Classifer
#clf = clf.fit(X_train,y_train)
##Predict the response for test dataset
#y_pred = clf.predict(X_test)
##counter = Counter(t_pred)
##print(counter)
##y_pred = clf.predict(X_test)
#confusion_matrix(y_test, y_pred, labels=[0, 1])
#accuracy_score(y_test, y_pred)
#############            DecisionTreeClassifier                          ##################
test1 = pd.read_csv('test.csv', sep=';')  
T,t = prepareTest(test)

test = test.dropna()

epoch = datetime.datetime.utcfromtimestamp(0)
test.loc[:,'TS'] = pd.to_datetime(test['TS'])
test.loc[:,'TS'] = test.loc[:,'TS'] - epoch
test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()
#da inserire il TS
T = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
T = T.to_numpy()
T = T[numpy.logical_not(numpy.isnan(T))]
test.loc[:,'VAR_CLASS'] = pd.Series(t_pred)

len(test[(test['VAR_CLASS']== 1) | (test['VAR_CLASS']== 2)]['KIT_ID'].unique())
len(test['KIT_ID'].unique())

kit113054467 = test[test['KIT_ID'] == 113054467]
kit113054467.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit113054467.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
kit113054467.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit113054467.plot(x='TS',y='AVG_SPEED_DW',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante


kit1338038850 = test[test['KIT_ID'] == 1338038850]
kit1338038850.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit1338038850.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
kit1338038850.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit1338038850.plot(x='TS',y='AVG_SPEED_DW',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante


kit2830968677 = test[test['KIT_ID'] == 2830968677]
kit2830968677.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit2830968677.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
kit2830968677.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit2830968677.plot(x='TS',y='AVG_SPEED_DW',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante

kit2830968677 = kit2830968677.dropna()

kit2830968677.loc[:,['USAGE','VAR_CLASS']] = preprocessing.normalize(kit2830968677.loc[:,['USAGE','VAR_CLASS']])


########################################Parte relatica alla lettura del TEST.csv
#test['TS'] = pd.to_numeric(test['TS'], downcast='float', errors='ignore')
#test = pd.read_csv('test.csv', sep=';')       



#pulire il training dai kitid con 1 e 2(
#        allenare il model con il nuovo dataset, riallenarlo con il primo kitid1, kitid2,kitid3)
#pulire il training dai kitid con 0 e allenare il modello
#pulire il training dai kitid con 0 e allenare il modello, e riallenarlo con il training pulito da 1 e 2





###################    STAGIONALITA' ##############################
epoch = datetime.datetime.utcfromtimestamp(0)
training = pd.read_csv('training.csv', sep=';')
training.loc[:,'TS'] = pd.to_datetime(training['TS'])
training.loc[:,'TS'] = training.loc[:,'TS'] - epoch
training.loc[:,'TS'] = training.loc[:,'TS'].dt.total_seconds()
#da inserire il TS
kit1629361016 = training[training['KIT_ID'] == 1629361016]
kit1629361016 = kit1629361016.loc[:,['TS','USAGE']]
kit1629361016 = kit1629361016.set_index('TS')
#kit1629361016.loc[:,'TS'] = pd.to_datetime(kit1629361016.loc[:,'TS'])
kit1629361016.plot()
type(kit1629361016.loc[885,'TS'])

X = kit1629361016.values
diff = list()
ore = 2160 # 24*60 /5 
for i in range(ore, len(X)):
	value = X[i] - X[i - ore]
	diff.append(value)
pyplot.plot(diff)
pyplot.show()

X = kit1629361016.values
diff = list()
days = 2016  ##288 * 7
for i in range(days, len(X)):
	value = X[i] - X[i - days]
	diff.append(value)
pyplot.plot(diff)
pyplot.show()


# fit polynomial: x^2*b1 + x*b2 + ... + bn
X = [i%288 for i in range(0, len(kit1629361016))]
y = kit1629361016.values
degree = 4
coef = np.polyfit(X, y, degree)
print('Coefficients: %s' % coef)
# create curve
curve = list()
for i in range(len(X)):
	value = coef[-1]
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
	curve.append(value)
# plot curve over original data
pyplot.plot(kit1629361016.values)
pyplot.plot(curve, color='red', linewidth=3)
pyplot.show()


pyplot.plot(kit1629361016.values)
len(X)





training.loc[:,'VAR_CLASS'] = training.loc[:,'VAR_CLASS'].replace(2,1) 
X,y = prepareTraining2(training)
counter = Counter(y)
print(counter)
#Synthetic Minority Over-sampling Technique
oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
counter = Counter(y_train)
print(counter)
counter = Counter(y_test)
print(counter)
results = Results()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
#    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print(score)
print(confusionMatrix)
results.set_accuracy_score(score)
results.set_confusion_matrix(confusionMatrix)
results.set_clf(clf)
return results