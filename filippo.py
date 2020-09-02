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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier 
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

print('VAR_CLASS 0: ' + str(len(training[training['VAR_CLASS'] == 0])))#16521526
print('VAR_CLASS 1: ' + str(len(training[training['VAR_CLASS'] == 1])))#36
print('VAR_CLASS 2: ' + str(len(training[training['VAR_CLASS'] == 2])))#472

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

training

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

##############
kitWith1or2.loc[:,'VAR_CLASS'] = kitWith1or2.loc[:,'VAR_CLASS'].replace(2,1)
###########






# CREO DIVERSI DATAFRAME PER KIT_ID

# kit_id1 3409364152 
# kit_id2 1629361016
# kit_id3 2487219358 

training_diff_kit = training[(training['KIT_ID']!=3409364152) &
                             (training['KIT_ID']!=1629361016) &
                             (training['KIT_ID']!=2487219358)]

training_kit_id1 = training[training.KIT_ID==3409364152]
training_kit_id2 = training[training.KIT_ID==1629361016]
training_kit_id3 = training[training.KIT_ID==2487219358]

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

X,y = prepareTraining2(training)
counter = Counter(y)
print(counter)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train








#ottengo var_class= 0 

kitWith1or2 = training[((training['KIT_ID'] == 3409364152) | (training['KIT_ID']== 1629361016) | (training['KIT_ID']== 2487219358))]
kitNot1or2 = training[((training['KIT_ID'] != 3409364152) & (training['KIT_ID']!= 1629361016) & (training['KIT_ID']!= 2487219358))]




dfmklsdffklsdfklsdfklòklòsdfklòdfjklsdf


#31/08/20



# CREO DIVERSI DATAFRAME PER KIT_ID e FACCIO LA PREPARE TRAINING PER KIT1

# kit_id1 3409364152 
# kit_id2 1629361016
# kit_id3 2487219358 

training_diff_kit = training[(training['KIT_ID']!=3409364152) &
                             (training['KIT_ID']!=1629361016) &
                             (training['KIT_ID']!=2487219358)]

training_kit_id1 = training[training.KIT_ID==3409364152]
training_kit_id2 = training[training.KIT_ID==1629361016]
training_kit_id3 = training[training.KIT_ID==2487219358]





print('0 ' + str(len(training_kit_id1[training_kit_id1['VAR_CLASS'] == 0])))#8480
print('1 ' + str(len(training_kit_id1[training_kit_id1['VAR_CLASS'] == 1])))#12
print('2 ' + str(len(training_kit_id1[training_kit_id1['VAR_CLASS'] == 2])))#140


PROVA2020= training_kit_id1.drop(training_kit_id1[training_kit_id1.VAR_CLASS >0].index)

print('0 ' + str(len(PROVA2020[PROVA2020['VAR_CLASS'] == 0])))#8480
print('1 ' + str(len(PROVA2020[PROVA2020['VAR_CLASS'] == 1])))#0
print('2 ' + str(len(PROVA2020[PROVA2020['VAR_CLASS'] == 2])))#0


#PROVA2020 SI RIFERISCE AL DF CHE PRESENTA SOLTANTO IL KIT1 CON SOLTANTO VAR_CLASS=0


#ORA PREPARO IL DATASET COME ABBIAMO FATTO FINO AD ORA

def prepareTraining2(PROVA2020):
    epoch = datetime.datetime.utcfromtimestamp(0)
    PROVA2020.loc[:,'TS'] = pd.to_datetime(PROVA2020['TS'])
    PROVA2020.loc[:,'TS'] = PROVA2020.loc[:,'TS'] - epoch
    PROVA2020.loc[:,'TS'] = PROVA2020.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = PROVA2020.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = PROVA2020.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)
X,y = prepareTraining2(PROVA2020)
counter = Counter(y)
print(counter)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train






#PROVO I VARI M0DELLI 

#logistic_regression
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




































#PROVE BINARIE
#PREPARAZIONE COLONNA 1=2
training.loc[:,'VAR_CLASS'] = training.loc[:,'VAR_CLASS'].replace(2,1)
training['VAR_CLASS'].value_counts() #0=16521526 1=508

#preprazione test e train
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

X,y = prepareTraining2(training)
counter = Counter(y)
print(counter)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train



#########################EVENTUALE BILANCIAMENTO 


####################################ALGORITMO 

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

#######risultati
print(format(logreg.score(X_test, y_test))) #accuracy 0.99999999999

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


print(classification_report(y_test, y_pred))





















#####################################################
# CREO DIVERSI DATAFRAME PER KIT_ID e FACCIO LA PREPARE TRAINING PER KIT1

# kit_id1 3409364152 
# kit_id2 1629361016
# kit_id3 2487219358 

training_diff_kit = training[(training['KIT_ID']!=3409364152) &
                             (training['KIT_ID']!=1629361016) &
                             (training['KIT_ID']!=2487219358)]

training_kit_id1 = training[training.KIT_ID==3409364152]
training_kit_id2 = training[training.KIT_ID==1629361016]
training_kit_id3 = training[training.KIT_ID==2487219358]





print('0 ' + str(len(training_kit_id1[training_kit_id1['VAR_CLASS'] == 0])))#8480
print('1 ' + str(len(training_kit_id1[training_kit_id1['VAR_CLASS'] == 1])))#12
print('2 ' + str(len(training_kit_id1[training_kit_id1['VAR_CLASS'] == 2])))#140


PROVA2020= training_kit_id1.drop(training_kit_id1[training_kit_id1.VAR_CLASS >0].index)

print('0 ' + str(len(PROVA2020[PROVA2020['VAR_CLASS'] == 0])))#8480
print('1 ' + str(len(PROVA2020[PROVA2020['VAR_CLASS'] == 1])))#0
print('2 ' + str(len(PROVA2020[PROVA2020['VAR_CLASS'] == 2])))#0





#LAVORIAMO CON PROVA2020
#ABBIAMO 8480 ROW CON VAR_CLASS SOLO UGUALE A 0
#♦OTTENIAMO CON SVC UN ACCURACY PARI A 0.893
#KIT_ID==3409364152


def prepareTraining2(PROVA2020):
    epoch = datetime.datetime.utcfromtimestamp(0)
    PROVA2020.loc[:,'TS'] = pd.to_datetime(PROVA2020['TS'])
    PROVA2020.loc[:,'TS'] = PROVA2020.loc[:,'TS'] - epoch
    PROVA2020.loc[:,'TS'] = PROVA2020.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = PROVA2020.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = PROVA2020.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)
X,y = prepareTraining2(PROVA2020)
counter = Counter(y)
print(counter)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train
len(X_train)
len(y_train)
counter = Counter(y_test)
print(counter)
X, y = make_classification(n_samples=2544, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = OneVsRestClassifier(SVC())
# fit model
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(X_test)
y_pred
SVC=accuracy_score(y_test, y_pred) #0.893 - #: 0.6615566037735849
SVC


# training a DescisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test)
 
dtree=accuracy_score(y_test, y_pred)
dtree





#LAVORO CON PROVA2021= moltiplico per 50 le righe di PROVA2020
#432480 row 
PROVA2021 = PROVA2020.sample(n=16501599, random_state=1234, replace=True)
#PROVA2021=PROVA2020.append([PROVA2020]*50,ignore_index=True)
PROVA2021



def prepareTraining3(PROVA2021):
    epoch = datetime.datetime.utcfromtimestamp(0)
    PROVA2021.loc[:,'TS'] = pd.to_datetime(PROVA2021['TS'])
    PROVA2021.loc[:,'TS'] = PROVA2021.loc[:,'TS'] - epoch
    PROVA2021.loc[:,'TS'] = PROVA2021.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = PROVA2021.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = PROVA2021.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)


X,y = prepareTraining3(PROVA2021)
counter = Counter(y)
print(counter)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train
len(X_train)
len(y_train)
counter = Counter(y_test)
print(counter)
X, y = make_classification(n_samples=2544, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = OneVsRestClassifier(SVC())
# fit model
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(X_test)
y_pred
SVC2021=accuracy_score(y_test, y_pred) #0.893 - #: 0.6615566037735849
SVC2021

#####################################################


#####################################
#logistic_regression
X, y = make_classification(n_samples=4950480, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=100)
# define model
model2 = LogisticRegression()
# define the ovr strategy
model3 = OneVsRestClassifier(model2)
# fit model
model3.fit(X_train, y_train)
# make predictions
y_pred = model3.predict(X_test)
y_pred

logistic_regression=accuracy_score(y_test, y_pred)#0.696 con random state=1 con radom state=100 -> 0.652
logistic_regression

#############################################

#MultinomialNB

X, y = make_classification(n_samples=4950480, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=100)
# define model
model2_b = MultinomialNB()

# define the ovr strategy
model3_b = OneVsRestClassifier(model2_b)
# fit model
model3_b.fit(X_train, y_train)
# make predictions
y_pred = model3_b.predict(X_test)
y_pred
MultinomialNB=accuracy_score(y_test, y_pred)
MultinomialNB

# training a DescisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test)
 
dtree=accuracy_score(y_test, y_pred)
dtree
########################dopo aver testato con solo var_class=0, proviamo con il df iniziale 

#MultinomialNB senza oversample



def prepareTraining4(training):
    epoch = datetime.datetime.utcfromtimestamp(0)
    training.loc[:,'TS'] = pd.to_datetime(training['TS'])
    training.loc[:,'TS'] = training.loc[:,'TS'] - epoch
    training.loc[:,'TS'] = training.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = training.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = training.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)

X,y = prepareTraining4(training)
counter = Counter(y)
print(counter)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train
len(X_train)
len(y_train)
counter = Counter(y_test)

print(counter)

#MultinomialNB

X, y = make_classification(n_samples=4950480, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=100)
# define model
model2_b = MultinomialNB()
# define the ovr strategy
model3_b = OneVsRestClassifier(model2_b)
# fit model
model3_b.fit(X_train, y_train)
# make predictions
y_pred = model3_b.predict(X_test)
y_pred
MultinomialNB=accuracy_score(y_test, y_pred)
MultinomialNB #0.4486539290656459


confusion= confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)



###########################PREPARAZIONE con onversample 


def prepareTraining4(training):
    epoch = datetime.datetime.utcfromtimestamp(0)
    training.loc[:,'TS'] = pd.to_datetime(training['TS'])
    training.loc[:,'TS'] = training.loc[:,'TS'] - epoch
    training.loc[:,'TS'] = training.loc[:,'TS'].dt.total_seconds()
    #da inserire il TS
    X = training.loc[:,['TS','KIT_ID','USAGE','NUM_CLI']]
    y = training.loc[:,'VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)



X,y = prepareTraining4(training)
counter = Counter(y)
print(counter)
oversample = SMOTE(random_state=100,k_neighbors=2)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train
len(X_train)
len(y_train)
counter = Counter(y_test)

print(counter)

#MultinomialNB

X, y = make_classification(n_samples=4950480, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=100)
# define model
model2_b = MultinomialNB()
# define the ovr strategy
model3_b = OneVsRestClassifier(model2_b)
# fit model
model3_b.fit(X_train, y_train)
# make predictions
y_pred = model3_b.predict(X_test)
y_pred
MultinomialNB=accuracy_score(y_test, y_pred)
MultinomialNB #0.4083610379293708 
confusion= confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)
###############################


#logistic_regression
X, y = make_classification(n_samples=4950480, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=100)
# define model
model2 = LogisticRegression()
# define the ovr strategy
model3 = OneVsRestClassifier(model2)
# fit model
model3.fit(X_train, y_train)
# make predictions
y_pred = model3.predict(X_test)
y_pred
logistic_regression=accuracy_score(y_test, y_pred)
logistic_regression #0.536046305648106
confusion= confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)
##################SVC
X, y = make_classification(n_samples=2544, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = OneVsRestClassifier(SVC())
# fit model
model.fit(X_train, y_train)
# make predictions
y_pred = model.predict(X_test)
y_pred
SVC=accuracy_score(y_test, y_pred) 
SVC
#ci mette una vita a runnare
confusion= confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)
########################################## Decision

# training a DescisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test)
 
dtree=accuracy_score(y_test, y_pred)
dtree #0.6281149428348497




print(Counter({0: 36, 1: 36, 2: 36}))
