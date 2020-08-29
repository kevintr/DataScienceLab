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

        
training = pd.read_csv('training.csv', sep=';')  
# verifica valori null all'interno del training
training = training.dropna()

len(training['KIT_ID'].unique())
training.groupby('KIT_ID')['KIT_ID'].count().unique()
counter = Counter(training['KIT_ID'])
print(counter)


training['TS'] = unix_time_millis(training['TS'])
training['TS'] = training['TS'] - epoch
training['TS'] = training['TS'].dt.total_seconds()


#df_time = pd.to_datetime(training['TS'])
#
# 
#
#second= (df_time.dt.hour*60+df_time.dt.minute)*60 + df_time.dt.second
#
# 
#
#second.head()
#second.tail()
#Trasformazione TS in datetime
training['TS'] = pd.to_datetime(training['TS'])
training['TS'] = training['TS'].dt.total_seconds()

t = datetime.datetime(2011, 10, 21, 0, 0)
time.mktime(t.timetuple())
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
    training = pd.read_csv('training.csv', sep=';')   
    epoch = datetime.datetime.utcfromtimestamp(0)
    training['TS'] = pd.to_datetime(training['TS'])
    training['TS'] = training['TS'] - epoch
    training['TS'] = training['TS'].dt.total_seconds()
    #da inserire il TS
    X = training[['TS','KIT_ID','USAGE','NUM_CLI']]
    y = training['VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)


X,y = prepareTraining()
#In terms of machine learning, Clf is an estimator instance, which is used to store model.
#We use clf to store trained model values, which are further used to predict value, based on the previously stored weights.

# Generate a synthetic imbalanced classification dataset
#Synthetic Minority Over-sampling Technique
oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)
plot(counter)

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
class OvoClassifierResults:

    def __init__(self, confusion_matrix,accuracy_score):
        ## private varibale or property in Python
        self.__confusion_matrix = confusion_matrix
        self.__accuracy_score = accuracy_score
        
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

#restituisce in outPut l'accuracy e la confusionMatrix per il classifier in input
def ovoClassifier(classifier):
    ovoClassifierResults = OvoClassifierResults()
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
#    confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
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



















##############                      PLOOOOOOOOOOOOOOOOOOOOOOT                    ############################################################
training2 = training[training['VAR_CLASS'] == 2]
training2['KIT_ID'].unique()# trovare gli unici KIT_ID che hanno avuto un disservizio di tipo 1

training1 = training[training['VAR_CLASS'] == 1]
training1['KIT_ID'].unique()# trovare gli unici KIT_ID che hanno avuto un disservizio di tipo 1

kit3409364152 = training.loc[(training.loc[:,'KIT_ID'] == 3409364152)]

training[training['AVG_SPEED_DW'] == 85320]['KIT_ID'].unique()

def normalizeSeries(seriesInDatframe):
    seriesInDatframe = (seriesInDatframe-seriesInDatframe.min())/(
           seriesInDatframe.max()-seriesInDatframe.min())
    return seriesInDatframe
    

kit3409364152.loc[:,'USAGE']= normalizeSeries(kit3409364152.loc[:,'USAGE'])
kit3409364152.loc[:,'VAR_CLASS']= normalizeSeries(kit3409364152.loc[:,'VAR_CLASS'])
kit3409364152.loc[:,'NUM_CLI']= normalizeSeries(kit3409364152.loc[:,'NUM_CLI'])

pyplot.figure(figsize=(20,3))
pyplot.plot(kit3409364152.loc[:,'TS'],kit3409364152.loc[:,'USAGE'],linewidth=1)
pyplot.scatter(kit3409364152.loc[:,'TS'],kit3409364152.loc[:,'VAR_CLASS'], color='red',linewidth=None,edgecolors=None , marker='o')
pyplot.plot(kit3409364152.loc[:,'TS'],kit3409364152.loc[:,'NUM_CLI'], color='c',linewidth=3)
pyplot.xticks(np.arange(min(kit3409364152['TS']), max(kit3409364152['TS'])+datetime.timedelta(days=1), datetime.timedelta(days=1)),rotation=30)
pyplot.legend(('USAGE', 'NUM_CLI', 'VAR_CLASS'))
pyplot.show()



t=kit3409364152.loc[:,'TS']
s1=kit3409364152.loc[:,'USAGE']
s2=kit3409364152.loc[:,'VAR_CLASS']
s3=kit3409364152.loc[:,'NUM_CLI']

ax1 = plt.subplot(311)
plt.plot(t, s1)
plt.setp(ax1.get_xticklabels(), fontsize=6)

# share x only
ax2 = plt.subplot(312, sharex=ax1)
plt.plot(t, s2)
# make these tick labels invisible
plt.setp(ax2.get_xticklabels(), visible=False)

# share x and y
ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot(t, s3)
plt.xlim(0.01, 5.0)
plt.show()

fig, ax = plt.subplots(3)
x= kit3409364152['TS'].to_numpy()
y= kit3409364152['USAGE'].to_numpy()
z= kit3409364152['NUM_CLI'].to_numpy()
w= kit3409364152['VAR_CLASS'].to_numpy()
 
ax[0].margins(2,2)
ax[1].margins(2,2)
ax[2].margins(2,2)
ax[0].plot(x,y)
ax[0].set_title('Zoomed in')
plt.xticks(np.arange(min(kit3409364152['TS']), max(kit3409364152['TS'])+datetime.timedelta(days=1), datetime.timedelta(days=2)))
ax[1].plot(x,z)
ax[1].set_title('NUM_CLI')
ax[2].plot(x,w)
#ax[0].figure(figsize=(20,50))
#plt.ylim(100, 100)
plt.show()

k = kit3409364152.plot(x='TS',y='VAR_CLASS',color='orange',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
#plt.xticks(np.arange(min(kit3409364152['TS']), max(kit3409364152['TS'])+datetime.timedelta(days=1), datetime.timedelta(days=2)))
kit3409364152.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
fig, ax = plt.subplots()
ax.plot(X,Y1,'o')
ax.plot(X,Y2,'x')
plt.show()


kit3409364152.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit3409364152.plot(x='TS',y='AVG_SPEED_DW',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
#kit3409364152.plot(x='TS',y='AVG_SPEED_DW',color='red')#costante
plt.show()

kit3409364152[kit3409364152['VAR_CLASS'] == 2]['AVG_SPEED_DW']
kit3409364152[kit3409364152['VAR_CLASS'] == 1]
kit3409364152[kit3409364152['VAR_CLASS'] == 0].tail(320)[['USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI']]

print(Counter(kit3409364152['VAR_CLASS'])) #Counter({0: 8480, 2: 140, 1: 12})

########################################################################
kit1629361016 = training[training['KIT_ID'] == 1629361016]
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()
fig.autofmt_xdate()
kit1629361016.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
kit1629361016.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)
kit1629361016.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)
plt.xlabel('Index Values')
plt.ylabel('Elements in List Y')
plt.show()

kit1629361016 = training[training['KIT_ID'] == 1629361016]

fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

fig.subplots_adjust(bottom=0.25)

ax1_pos = fig.add_axes([0.2, 0.1, 0.65, 0.03])
ax2_pos = fig.add_axes([0.2, 0.05, 0.65, 0.03])

s1 = Slider(ax1_pos, 'Pos1', 0.1, 1000)
s2 = Slider(ax2_pos, 'Pos2', 0.1, 1000)

def update1(v):
    pos = s1.val
    ax1.axis([pos,pos+2,0,1])
    fig.canvas.draw_idle()

def update2(v):
    pos = s2.val
    ax2.axis([pos,pos+2,0,1])
    fig.canvas.draw_idle()

s1.on_changed(update1)
s2.on_changed(update2)
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()
fig.autofmt_xdate()
ax1.plot(kit1629361016['TS'],kit1629361016['VAR_CLASS'],'b-')
ax2.plot(kit1629361016['TS'],kit1629361016['USAGE'],'r-')
plt.show()

#kit1629361016[['USAGE', 'VAR_CLASS']][:10000].plot(x='TS',figsize=(15,2.5), linewidth=1, fontsize=10)
#kit1629361016.plot(x='TS',y='AVG_SPEED_DW',color='red',figsize=(20,10), linewidth=5, fontsize=5)#costante
#kit1629361016.plot(x='TS',y='NUM_CLI',color='red',figsize=(20,10), linewidth=5, fontsize=5)#costante
kit1629361016.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
kit1629361016.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)
kit1629361016.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)
#plt.xlabel('TS', fontsize=10);
#plt.ylabel('USAGE in bit/s', fontsize=10);

kit1629361016[kit1629361016['VAR_CLASS'] == 2]
kit1629361016[kit1629361016['VAR_CLASS'] == 1]
kit1629361016[kit1629361016['VAR_CLASS'] == 0].tail(335)[['TS','USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI']]
kit1629361016[kit1629361016['TS'] == '2018-11-28 20:50:00']
print(Counter(kit1629361016['VAR_CLASS']))#({0: 8024, 2: 312, 1: 12})

########################################################################
kit2487219358 = training[training['KIT_ID'] == 2487219358]
#kit2487219358.plot(x='TS',y='AVG_SPEED_DW',color='red')#costante
kit2487219358.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
kit2487219358.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit2487219358.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
kit2487219358.plot(x='TS',y='AVG_SPEED_DW',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
plt.show()

kit2487219358[kit2487219358['VAR_CLASS'] == 2]
kit2487219358[kit2487219358['VAR_CLASS'] == 1]
kit2487219358[kit2487219358['VAR_CLASS'] == 0].tail(320)[['USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI']]

print(Counter(kit2487219358['VAR_CLASS']))#({0: 3423, 2: 20, 1: 12})




#'''
#Split train/test from any given data point.
#:parameter
#    :param ts: pandas Series
#    :param test: num or str - test size (ex. 0.20) or index position
#                 (ex. "yyyy-mm-dd", 1000)
#:return
#    ts_train, ts_test
#'''
#def split_train_test(ts, test=0.20, plot=True, figsize=(15,5)):
#    ## define splitting point
#    if type(test) is float:
#        split = int(len(ts)*(1-test))
#        perc = test
#    elif type(test) is str:
#        split = ts.reset_index()[ 
#                      ts.reset_index().iloc[:,0]==test].index[0]
#        perc = round(len(ts[split:])/len(ts), 2)
#    else:
#        split = test
#        perc = round(len(ts[split:])/len(ts), 2)
#    print("--- splitting at index: ", split, "|", 
#          ts.index[split], "| test size:", perc, " ---")
#    
#    ## split ts
#    ts_train = ts.head(split)
#    ts_test = ts.tail(len(ts)-split)
#    if plot is True:
#        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, 
#                               sharey=True, figsize=figsize)
#        ts_train.plot(ax=ax[0], grid=True, title="Train", 
#                      color="black")
#        ts_test.plot(ax=ax[1], grid=True, title="Test", 
#                     color="black")
#        ax[0].set(xlabel=None)
#        ax[1].set(xlabel=None)
#        plt.show()
#        
#    return ts_train, ts_test
#
#
#kit2487219358_train , kit2487219358_test = split_train_test(kit2487219358[['TS','USAGE']])
#
#plot(kit2487219358[['TS','USAGE']], plot_ma=True, plot_intervals=True, window=w, figsize=(15,5))









####################################    PROVA PREDIZIONE CON SOLO KIT_DI CON 1E 2 #######################


#kitWith1or2 = training[(training['KIT_ID'] == 3409364152)]
kitWith1or2 = training[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
kitWith1or2.loc[:,'VAR_CLASS'] = kitWith1or2.loc[:,'VAR_CLASS'].replace(2,1)


def prepareTraining2(training):
    epoch = datetime.datetime.utcfromtimestamp(0)
    training.loc[:,'TS'] = pd.to_datetime(training['TS'])
    training.loc[:,'TS'] = training.loc[:,'TS'] - epoch
    training.loc[:,'TS'] = training.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = training.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y =  training.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)

X,y = prepareTraining2(kitWith1or2)
counter = Counter(y)
print(counter)



len(X_train)
len(y_train)
counter = Counter(y_test)
print(counter)

#Synthetic Minority Over-sampling Technique
oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train

resultLinearSVR = ovoClassifier(LinearSVR())
resultLinearSVC = ovoClassifier(LinearSVC())#----------
resultNuSVR = ovoClassifier(NuSVC())
#resultLinearSVR = ovoClassifier(NuSVR())
resultOneClassSVM = ovoClassifier(OneClassSVM())######tiene in conto tutti i valori
resultSVC = ovoClassifier(SVC())########
resultSVR = ovoClassifier(SVR())#####
resultSVR = ovoClassifier(LogisticRegression())#######

clf = OneVsRestClassifier(classifier)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print(score)
print(confusionMatrix)

























############    Random Forest ##################


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
confusion_matrix(y_test, y_pred, labels=[0, 1])
accuracy_score(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
############    Random Forest ##################

############            DecisionTreeClassifier                          ##################OTTIMI RISULATI
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
t_pred = clf.predict(T)
counter = Counter(t_pred)
print(counter)
#y_pred = clf.predict(X_test)
confusion_matrix(w, y_pred, labels=[0, 1])
accuracy_score(y_test, y_pred)
############            DecisionTreeClassifier                          ##################




kitNot1or2 = training[((training['KIT_ID'] != 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]
kitNot1or2[kitNot1or2['VAR_CLASS']==1]


kitNot1or2 = kitNot1or2.loc[:,'TS','KIT_ID','USAGE','NUM_CLI']
kitNot1or2 = kitNot1or2.loc[:,'VAR_CLASS']


Z,w = prepareTraining2(kitNot1or2)
Z_train, Z_test, w_train, w_test = train_test_split(Z, w, test_size=0.3, random_state=100)
w_pred = clf.predict(T)
counter = Counter(w_pred)
print(counter)
confusion_matrix(w, w_pred, labels=[0, 1])
accuracy_score(w, w_pred)



test = pd.read_csv('test.csv', sep=';')  
T,t = prepareTraining2(test)

test = test.dropna()

epoch = datetime.datetime.utcfromtimestamp(0)
test.loc[:,'TS'] = pd.to_datetime(training['TS'])
test.loc[:,'TS'] = test.loc[:,'TS'] - epoch
test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()
#da inserire il TS
T = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
T = T.to_numpy()
test.loc[:,'VAR_CLASS'] = pd.Series(t_pred)

len(test[test['VAR_CLASS']== 1]['KIT_ID'].unique())
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
days = 288 # 24*60 /5 
for i in range(days, len(X)):
	value = X[i] - X[i - days]
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
X = [i%269 for i in range(0, len(kit1629361016))]
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





