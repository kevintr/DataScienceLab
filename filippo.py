from sklearn.datasets import make_classification
from datetime import datetime
from datetime import time,date
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.under_sampling import NearMiss
import pandas as pd
import numpy as np
import datetime as dt
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
from babel.dates import format_date, format_datetime


training =pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")    
#trainingCampionato = training.sample(n=1600 , replace=False,random_state=1000) #['VAR_CLASS']

varclass_0= training.drop(training[training.VAR_CLASS >0].index)
#SOLO CON KIT TRA PARANTESI
training_diff_kit = training[(training['KIT_ID']==3409364152) |
                             (training['KIT_ID']==1629361016) |
                             (training['KIT_ID']==2487219358)]

trainingCampionato = varclass_0.sample(n=100000,replace=False,random_state=100)
FINALE= pd.concat([training_diff_kit,trainingCampionato])
#SENZA KIT TRA PARENTESI
training_VAR0 = training[(training['KIT_ID']!=3409364152) &
                             (training['KIT_ID']!=1629361016) &
                             (training['KIT_ID']!=2487219358)]

trainingCampionato2 = training_VAR0.sample(n=100000,replace=False,random_state=100)
FINALE2= pd.concat([training_diff_kit,trainingCampionato2])



# ULTIMATE = pd.merge(trainingCampionato, training_diff_kit[['TS','USAGE','NUM_CLI','VAR_CLASS','KIT_ID']], on='KIT_ID')

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
#                  4.GaussianNB



#GAUSS TOP PROVA (PIU' E' VICINO A ZERO MEGLIO E')




#MODELLI USATI 
ALGORITMO_RISULTATI(FINALE,prepareTraining_oversmaple_campionato,GaussiannB)#0.4811
ALGORITMO_RISULTATI(FINALE,prepareTraining_oversmaple_campionato,RANDOMFOREST)#1.000
ALGORITMO_RISULTATI(FINALE,prepareTraining_oversmaple_campionato,Logisticregression)#0.5283
ALGORITMO_RISULTATI(FINALE,prepareTraining_oversmaple_campionato,Multinomialnb)#0.4110
ALGORITMO_RISULTATI(FINALE,prepareTraining_oversmaple_campionato,SVCmodel)
ALGORITMO_RISULTATI(FINALE,prepareTraining_oversmaple_campionato,MLPmodel)#0.7482


#MODELLI USATI 2
ALGORITMO_RISULTATI(FINALE2,prepareTraining_oversmaple_campionato,GaussiannB) #0.4816
ALGORITMO_RISULTATI(FINALE2,prepareTraining_oversmaple_campionato,RANDOMFOREST)#1.0000
ALGORITMO_RISULTATI(FINALE2,prepareTraining_oversmaple_campionato,Logisticregression)#0.5267
ALGORITMO_RISULTATI(FINALE2,prepareTraining_oversmaple_campionato,Multinomialnb)#0.4092
ALGORITMO_RISULTATI(FINALE2,prepareTraining_oversmaple_campionato,SVCmodel)
ALGORITMO_RISULTATI(FINALE2,prepareTraining_oversmaple_campionato,MLPmodel) #0.7713

#TEST MODELLI
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,GaussiannB)
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,RANDOMFOREST)
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,Logisticregression)
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,Multinomialnb)
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,Decision_Tree)
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,SVCmodel)
ALGORITMO_RISULTATI(FINALE,prepareTraining_senza1o2,MLPmodel)

ALGORITMO_RISULTATI(training,prepareTraining_oversmaple_campionato,MLPmodel)



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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
    return(X_train, X_test, y_train, y_test)




def prepareTraining_oversmaple_campionato(dataset):
    dataset= dataset.dropna()
    
    epoch = datetime.datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    # dataset1 =dataset.sample(n=600 , replace=False,random_state=1000) 
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    oversample = SMOTE(random_state=100,k_neighbors=2)
    X, y = oversample.fit_resample(X, y)
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100,stratify=y)
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
    counter_ypred= Counter(y_pred)
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix,counter_ypred)
#Logisticregression

def Logisticregression(X_train, X_test, y_train, y_test):
    model_logistic= LogisticRegression()
    logistic_model=OneVsRestClassifier(model_logistic)
    modello_allenato=logistic_model.fit(X_train, y_train) 
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
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    counter_ypred= Counter(y_pred)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix)

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
    counter_ypred= Counter(y_pred)
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix,counter_ypred)

def GaussiannB(X_train, X_test, y_train, y_test):
    model_nb= GaussianNB()
    # nb_model=OneVsRestClassifier(model_nb)
    nb_model=model_nb
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
    counter_ypred= Counter(y_pred)
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix,counter_ypred)


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
    counter_ypred= Counter(y_pred)
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix,counter_ypred)



def SVCmodel(X_train,X_test, y_train, y_test):
    model= OneVsRestClassifier(SVC())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    counter_ypred= Counter(y_pred)
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
# Specificity or true negative rate
    TNR = TN/(TN+FP) 
# Precision or positive predictive value
    PPV = TP/(TP+FP)
# Negative predictive value
    NPV = TN/(TN+FN)
# Fall out or false positive rate
    FPR = FP/(FP+TN)
# False negative rate
    FNR = FN/(TP+FN)
# False discovery rate
    FDR = FP/(TP+FP)
# Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix,counter_ypred)




#https://medium.com/@b.terryjack/tips-and-tricks-for-multi-class-classification-c184ae1c8ffc
def MLPmodel(X_train,X_test, y_train, y_test):
    model= MLPClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    counter_ypred= Counter(y_pred)
    cnf_matrix=confusion_matrix(y_test, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
# Specificity or true negative rate
    # TNR = TN/(TN+FP) 
# Precision or positive predictive value
    # PPV = TP/(TP+FP)
# Negative predictive value
    # NPV = TN/(TN+FN)
# Fall out or false positive rate
    # FPR = FP/(FP+TN)
# False negative rate
    # FNR = FN/(TP+FN)
# False discovery rate
    # FDR = FP/(TP+FP)
# Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix,counter_ypred,TPR,ACC)








test = pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\test.csv",";" )
test
test["VAR_ClASS"] = ""
test

test
test = test.dropna()
test
np.nan
#Trasformazione TS in datetime 
# test['TS'] = pd.to_datetime(test['TS']) 
# test.loc[0,'TS'] 

def prepareTraining1(test):       
    epoch = datetime.utcfromtimestamp(0)     
    test.loc[:,'TS'] = pd.to_datetime(test['TS'])     
    test.loc[:,'TS'] = test.loc[:,'TS'] - epoch     
    test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()         
    T_1 = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    T_2 = test.loc[:,'VAR_CLASS']
    T_1= T_1.to_numpy()   
    T_2= T_2.to_numpy()
    return T_1,T_2 

def prepareTrainingVSTest(dataset):
    # test = test.dropna()
    test['TS'] = pd.to_datetime(test['TS']) 
    test.loc[0,'TS'] 
    dataset= dataset.dropna()
    epoch = datetime.utcfromtimestamp(0)
    dataset.loc[:,'TS'] = pd.to_datetime(dataset['TS'])
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'] - epoch
    dataset.loc[:,'TS'] = dataset.loc[:,'TS'].dt.total_seconds()
    X = dataset.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = dataset.loc[:,'VAR_CLASS']
    X = X.to_numpy()
    y = y.to_numpy()
    T_1,T_2=prepareTraining1(test)
    X_train, NO, y_train, no= train_test_split(X,y)
    # YES, X_test, yes, y_test= train_test_split(T_1,T_2)
    X_test=T_1
    y_test=T_2
    return(X_train, X_test, y_train, y_test)

def Logisticregression(X_train, X_test, y_train, y_test):
    model_logistic= LogisticRegression()
    logistic_model=OneVsRestClassifier(model_logistic)
    logistic_model.fit(X_train, y_train) 
    # modello_allenato= logistic_model.fit(X_train, y_train) 
    y_pred = logistic_model.predict(X_test)
    # y_pred = logistic_model.predict(y_test)
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
    Confusion_matrix= confusion_matrix(y_test, y_pred)
    counter_ypred= Counter(y_pred)
    return(b,c,d,e,f,g,h,i,l,m,Confusion_matrix)


prepareTrainingVSTest(FINALE)


ALGORITMO_RISULTATI(FINALE2,prepareTrainingVSTest,Logisticregression)
ALGORITMO_RISULTATI(FINALE2,prepareTrainingVSTest,MLPmodel)


test.loc[:,'VAR_CLASS'] = pd.Series(X_test) 
len(test[(test['VAR_CLASS']== 1)| (test['VAR_CLASS']== 2)]['KIT_ID'].unique())
len(test['KIT_ID'].unique())

# (X_train, X_test, y_train, y_test):
# model_logistic= LogisticRegression()
# logistic_model=OneVsRestClassifier(model_logistic)
# logistic_model.fit(X_train, y_train) 
# y_pred = logistic_model.predict(X_test)









##############PROVE
# test = pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\test",";" )
test = test.dropna()


X_train, X_test, y_train, y_test=prepareTraining_oversmaple(FINALE)
model_logistic= LogisticRegression(X_train, X_test, y_train, y_test)
logistic_model=OneVsRestClassifier(model_logistic)
logistic_model.fit(X_train, y_train) 
y_pred = logistic_model.predict(X_test)


#Trasformazione TS in datetime 
test['TS'] = pd.to_datetime(test['TS']) 
test.loc[0,'TS'] 

# def prepareTraining50(test):      
#     epoch = datetime.utcfromtimestamp(0)     
#     test.loc[:,'TS'] = pd.to_datetime(test['TS'])     
#     test.loc[:,'TS'] = test.loc[:,'TS'] - epoch     
#     test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()         
#     T = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]     
#     T = T.to_numpy()     
#     return T 

def prepareTraining50(test):      
    # epoch = datetime.utcfromtimestamp(0)     
    test.loc[:,'TS'] = pd.to_datetime(test['TS'])     
    test.loc[:,'TS'] = test.loc[:,'TS']   
    test.loc[:,'TS'] = test.loc[:,'TS'].dt.total_seconds()         
    T = test.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]     
    T = T.to_numpy()     
    return T 


T = prepareTraining50(test) 
t_pred = logistic_model.predict(T) 
test.loc[:,'VAR_CLASS'] = pd.Series(t_pred) 
len(test[(test['VAR_CLASS']== 1) | (test['VAR_CLASS']== 2)]['KIT_ID'].unique()) # RF ne predice 2!! # DT ne predice 3!! 
len(test['KIT_ID'].unique()) #2121 test[test['VAR_CLASS'] == 2]