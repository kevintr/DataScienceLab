from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

training =pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")    



#  METODO -> ALGORITMO_RISULTATI(dataframe,preparazione,model)
# Utilizzo del metodo:
#     ALGORITMO_RISULTATI(dataframe,preparazione,model)
#     preparazioni possibili:
#                  1.prepareTraining_oversmaple (test e train sul training con oversample)
#                  2.prepareTraining_senza1o2 (test= solo var_class=0, train=normale dataset)
#                  3. prepareTraining2(test e train sul training senza oversample)
#                  4. prepareTraining_kit123vstraining (train dataset normale e test kit123)
#    scelta algoritmo:
#                  1.Decision_Tree
#                  2.Multinomialnb
#                  3.Logisticregression
#                  4. GaussianNB



#GAUSS TOP PROVA (PIU' E' VICINO A ZERO MEGLIO E')


ALGORITMO_RISULTATI(training,prepareTraining2,Decision_Tree)
ALGORITMO_RISULTATI(training,prepareTraining2,Multinomialnb)
ALGORITMO_RISULTATI(training,prepareTraining_senza1o2,GaussiannB)
ALGORITMO_RISULTATI(training,prepareTraining_oversmaple,GaussiannB)
ALGORITMO_RISULTATI(training,prepareTraining_kit123vstraining,GaussiannB)#TRAIN=traininig TEST=KI123
ALGORITMO_RISULTATI(training,prepareTraining_oversmaple ,GaussiannB)
ALGORITMO_RISULTATI(training,prepareTraining_senza1o2,RANDOMFOREST)
ALGORITMO_RISULTATI(training,prepareTraining_oversmaple ,RANDOMFOREST)



#ALGORITMO DEL RISULTATO

def ALGORITMO_RISULTATI(dataframe,preparazione,model):
    X_train, X_test, y_train, y_test= preparazione(dataframe)
    Risultati=model(X_train, X_test, y_train, y_test)
    return Risultati





#############metodi per la preparazione
#preparazione base 
def prepareTrainingEXAM (dataset):
    dataset= dataset.dropna()
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    X_1 = X.to_numpy()
    y_1 = y.to_numpy()
    return (X_1,y_1)

#preparazione dataset con oversample

def prepareTraining_oversmaple(dataset):
    dataset= dataset.dropna()
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    return(X_train, X_test, y_train, y_test)


#preparazione dataset con oversample e campione n=1000
def prepareTraining_senza1o2(dataset):
    varclass_0= dataset.drop(dataset[dataset.VAR_CLASS >0].index)
    dataset= dataset.dropna()
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    
    
    X = X.to_numpy()
    y = y.to_numpy()
    X, y = make_classification(n_samples=1000, n_features=4, random_state=100)
    X_1,y_1=prepareTrainingEXAM(varclass_0)
    X_train, NO, y_train, no= train_test_split(X,y)
    YES, X_test, yes, y_test= train_test_split(X_1,y_1)
    return(X_train, X_test, y_train, y_test)





def prepareTraining_kit123vstraining(dataset):
    dataset= dataset.dropna()
    kit123 = dataset[(dataset['KIT_ID']!=3409364152) &
                             (dataset['KIT_ID']!=1629361016) &
                             (dataset['KIT_ID']!=2487219358)]
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    dataset= dataset.dropna()
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    
    
    X = X.to_numpy()
    y = y.to_numpy()
    X, y = make_classification(n_samples=1000, n_features=4, random_state=100)
    X_1,y_1=prepareTrainingEXAM(kit123)
    X_train, NO, y_train, no= train_test_split(X,y)
    YES, X_test, yes, y_test= train_test_split(X_1,y_1)
    return(X_train, X_test, y_train, y_test)



#preparazione dataset senza oversample
def prepareTraining2(dataset):
    dataset= dataset.dropna()
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    return(X_train, X_test, y_train, y_test)


def prepareTraining_kit1_2_3(dataset):
    dataset= dataset.dropna()
    dataset = dataset[(dataset['KIT_ID']!=3409364152) &
                             (dataset['KIT_ID']!=1629361016) &
                             (dataset['KIT_ID']!=2487219358)]
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    return(X_train, X_test, y_train, y_test)




#metodi per gli algo

#decision tree
def Decision_Tree(X_train, X_test, y_train, y_test):
    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
    y_pred = dtree_model.predict(X_test)
    b= print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))
    c= print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    d= print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    e= print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    f= print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    g= print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    h=print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    i= print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    l= print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    m=print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
    return(b,c,d,e,f,g,h,i,l,m)
#Logisticregression

def Logisticregression(X_train, X_test, y_train, y_test):
    model_logistic= LogisticRegression()
    logistic_model=OneVsRestClassifier(model_logistic)
    logistic_model.fit(X_train, y_train) 
    y_pred = logistic_model.predict(X_test)
    b= print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))
    c= print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    d= print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    e= print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    f= print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    g= print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    h=print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    i= print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    l= print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    m=print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
    return(b,c,d,e,f,g,h,i,l,m)

#
def Multinomialnb(X_train, X_test, y_train, y_test):
    model_nb= MultinomialNB()
    nb_model=OneVsRestClassifier(model_nb)
    nb_model.fit(X_train, y_train) 
    y_pred = nb_model.predict(X_test)
    b= print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))
    c= print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    d= print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    e= print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    f= print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    g= print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    h=print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    i= print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    l= print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    m=print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
    return(b,c,d,e,f,g,h,i,l,m)

def GaussiannB(X_train, X_test, y_train, y_test):
    model_nb= GaussianNB()
    nb_model=OneVsRestClassifier(model_nb)
    nb_model.fit(X_train, y_train) 
    y_pred = nb_model.predict(X_test)
    b= print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))
    c= print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    d= print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    e= print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    f= print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    g= print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    h=print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    i= print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    l= print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    m=print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
    return(b,c,d,e,f,g,h,i,l,m)


def RANDOMFOREST(X_train,X_test, y_train, y_test):
    RF= RandomForestClassifier()
    RF_Model=OneVsRestClassifier(RF)
    RF_Model.fit(X_train, y_train)
    y_pred=RF_Model.predict(X_test)         
    b= print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_test, y_pred)))
    c= print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    d= print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    e= print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    f= print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    g= print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    h=print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    i= print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    l= print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    m=print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
    return(b,c,d,e,f,g,h,i,l,m)