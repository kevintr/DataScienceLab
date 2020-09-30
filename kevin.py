# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
@author:  Kevin Tranchina ,Filippo Maria Casula ,Giulia , Enrico Ragusa
"""
import pandas as pd
import numpy as np
import os #os è un libreria per induviduare la directory dove ci si trova
from datetime import datetime, timedelta 
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import *
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
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import preprocessing
import datetime, time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import make_classification as make_classification
from plotDsLab import plotUsageAndNumcliAndVarClassByTS
import methodsForAnalysisTrainingTest
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



#read file csv using PANDAS
training = pd.read_csv('training.csv', sep=';')  
# drop null value in training.csv
training = training.dropna()
# transformation TS in datetime
training['TS'] = pd.to_datetime(training['TS'])

print('0 ' + str(len(training[training['VAR_CLASS'] == 0])))#16521526 items with VAR_CLASS = 0 disservice 's absence 
print('1 ' + str(len(training[training['VAR_CLASS'] == 1])))#36 items with VAR_CLASS = 1 potential presence of disservice
print('2 ' + str(len(training[training['VAR_CLASS'] == 2])))#472 items with VAR_CLASS =2 disservice 


#Analysis column TS remove outliers
training['TS'] = pd.to_datetime(training['TS'])
training['TS'].dt.year.unique()#2018
training['TS'].dt.month.unique()#11
training['TS'].dt.day.unique()#1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
training['VAR_CLASS'].unique()#1,2,3
training[training['NUM_CLI'] >= 100]

#check column KIT_ID using describe()
training['KIT_ID'].describe()

#count    1.652203e+07
#mean     2.160265e+09
#std      1.224466e+09
#min      1.512491e+06
#25%      1.127272e+09
#50%      2.135241e+09
#75%      3.242074e+09
#max      4.292018e+09

#In terms of machine learning, Clf is an estimator instance, which is used to store model.
#We use clf to store trained model values, which are further used to predict value, based on the previously stored weights.

# Generate a synthetic imbalanced classification dataset
#Synthetic Minority Over-sampling Technique
#########################OneVsRestClassifier############################################
#al posto di clf possiamo mettere qualsiasi altra roba
clf = OneVsRestClassifier(SVC()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred) #0.2


#########################OneVsOneClassifier############################################
clf = OneVsOneClassifier(SVC()).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_test, y_pred) #0.2

#This approach is commonly used for algorithms that naturally predict numerical class 
#membership probability or score, such as: 
#Perceptron
#As such, the implementation of these algorithms in the scikit-learn library implements the OvR strategy 
#by default when using these algorithms for multi-class classification.
####################################    TRY PREDICTION WITH ONLY KIT_ID WITH 1 and 2 #######################
#training e test
kitWith1or2 = training.loc[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
# overlapping 2 with 1 in order to trasform problem in binary problem
kitWith1or2.loc[:,'VAR_CLASS'] = kitWith1or2.loc[:,'VAR_CLASS'].replace(2,1)

###### HOLD-OUT  #########################################################################################################
resultRandomForest = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(RandomForestClassifier(n_estimators=100),kitWith1or2)
adaBoostClassifier = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(AdaBoostClassifier(),kitWith1or2)
resultDecisionTree = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(DecisionTreeClassifier(),kitWith1or2)

###### Cross Validation ########
resultRandomForest = methodsForAnalysisTrainingTest.binaryCrossValidationClassifierSmote(RandomForestClassifier(n_estimators=100),kitWith1or2)
adaBoostClassifier = methodsForAnalysisTrainingTest.binaryCrossValidationClassifierSmote(AdaBoostClassifier(),kitWith1or2)
resultDecisionTree = methodsForAnalysisTrainingTest.binaryCrossValidationClassifierSmote(DecisionTreeClassifier(),kitWith1or2) 


################ more data ############################################################################################################
kitNot1or2 = training.loc[((training['KIT_ID']!= 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]

########## sampling #####################################################à
kitNot1or21HalfMilion =  kitNot1or2.sample(n=500000,replace=False)

frames = [kitNot1or21HalfMilion, kitWith1or2]
result = pd.concat(frames)
resultRandomForest = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(RandomForestClassifier(n_estimators=100),result)
adaBoostClassifier = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(AdaBoostClassifier(),result)
resultDecisionTree = methodsForAnalysisTrainingTest.binaryHoldOutClassifierSmote(DecisionTreeClassifier(),result)


############ PLOT of 3 KIT_ID whose var_char is 1 or 2 #############################################################
kit3409364152 = training.loc[training['KIT_ID'] == 3409364152]
kit1629361016 = training.loc[training['KIT_ID']== 1629361016]
kit2487219358 = training.loc[training['KIT_ID']== 2487219358]

###### Random Forest ##############
methodsForAnalysisTrainingTest.plotPredKitID(kit3409364152,resultRandomForest.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit1629361016,resultRandomForest.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit2487219358,resultRandomForest.get_clf())### in  input one kit_ID e one model trained 

###### Adaptive Boosting 1 ##############
methodsForAnalysisTrainingTest.plotPredKitID(kit3409364152,adaBoostClassifier.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit1629361016,adaBoostClassifier.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit2487219358,adaBoostClassifier.get_clf())### in  input one kit_ID e one model trained 
 
###### Decison Tree Classifier 1 ##############
methodsForAnalysisTrainingTest.plotPredKitID(kit3409364152,resultDecisionTree.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit1629361016,resultDecisionTree.get_clf())### in  input one kit_ID e one model trained 
methodsForAnalysisTrainingTest.plotPredKitID(kit2487219358,resultDecisionTree.get_clf())### in  input one kit_ID e one model trained 


#############            APPLICATION TO TEST.CSV                          ##################
test = pd.read_csv('test.csv', sep=';')

#prepare test to analisys
T= methodsForAnalysisTrainingTest.prepareTest(test)

y_pred = resultRandomForest.get_clf().predict(T)
np.unique(y_pred)
test.loc[:,'VAR_CLASS'] = pd.Series(y_pred)

test.loc[:,'VAR_CLASS'].value_counts()

len(test[(test['VAR_CLASS']== 1) | (test['VAR_CLASS']== 2)]['KIT_ID'].unique())## 377
len(test['KIT_ID'].unique())## 2121

test = methodsForAnalysisTrainingTest.fromSecondToDate(test)
kit1629361016 = test.loc[test.loc[:,'KIT_ID'] == 1629361016]# present var_class=0 even when usage is different to zero 
kit1970831019 = test[test['KIT_ID'] == 1970831019]
kit2709380104 = test[test['KIT_ID'] == 2709380104]
kit2130171679 = test[test['KIT_ID'] == 2130171679]
kit1824349749 = test[test['KIT_ID'] == 1824349749]# present var_class=0 even when usage is different to zero 

plotUsageAndNumcliAndVarClassByTS(kit1629361016,False)
########################## END ####################################################################################


"""
Spyder Editor Kevin Tranchina ,Filippo Maria Casula ,Giulia , Enrico Ragusa

This is a temporary script file.
"""