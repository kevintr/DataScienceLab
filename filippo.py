 #sono presenti prove e documentazione relativa al oneVSrest/random/vari plot
#sono prove, molte non funzionanti e altre su cui devo lavorare ancora

import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
training= pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")
training

training_test=training.groupby(['VAR_CLASS']).size().reset_index(name='counts')
training_test.head()
training_test.plot(kind='bar',x='VAR_CLASS',y='counts')
training_test.plot(x='VAR_CLASS',y='counts')
training_test.plot(kind='hist',x='VAR_CLASS',y='counts')
training_test.plot(kind='kde',x='VAR_CLASS',y='counts')
#####
#NON FUNZIONA 
#LINK: https://www.nintyzeros.com/2017/05/plotting-with-python-matplotlib-pandas-dataframe.html
df2 = training_test(np.random.rand(10, 4), columns=['VAR_CLASS'])
df2.plot.bar(stacked=True)
plt.show()
#####

####

training_test.apply(pd.to_numeric)


VAR_CLASS= training_test(['VAR_CLASS'])
COUNT= training_test(['counts'])

plt.plot(VAR_CLASS,COUNT)

######
#PROVA ONEvsREST..
#LINK:https://www.programcreek.com/python/example/94869/sklearn.multiclass.OneVsRestClassifier





# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##from subprocess import check_output
##print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


#import pandas as pd
#import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier


#train = pd.read_csv('../input/train.csv')


#train.info()

#train.describe()
#train.comment_text.head()

#creating x and y
x=training_test[:,'VAR_CLASS']
y = training_test(['counts'],axis=1)

#tokens on alphanumeric
tks = '[A-Za-z0-9]+(?=\\s+)'



# creating pipe line to fit 
#Pipelines help a lot when trying different cominations
pl = Pipeline([
        ('vec', CountVectorizer(token_pattern = tks)),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(x,y)

test = pd.read_csv('../input/test.csv')
test.info()
#1 missing value


test = test.fillna("")
#predicting
predictions = pl.predict_proba(test.comment_text)

# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=y.columns,
                             index=test.id,
                             data=predictions)


# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')







#########################################
import numpy as np
import pandas as pd

df_train = pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")

target_count = df_train.target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Class 2:', target_count[2])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
print('Proportion:', round(target_count[0] / target_count[2], 2), ': 1')
print('Proportion:', round(target_count[1] / target_count[2], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');






#######################################
# import the data sciecne libraries.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the dataset
data = df_train

# print the count of each class from the target vatiables
print(data.VAR_CLASS.value_counts())

# plot the count of each class from the target vatiables
sns.countplot(data.VAR_CLASS)

# import the function to compute the class weights
from sklearn.utils import compute_class_weight

# calculate the class weight by providing the 'balanced' parameter.
class_weight = compute_class_weight('balanced', data.VAR_CLASS.unique() , data.VAR_CLASS)

# print the result
print(class_weight)

#########################


from imblearn.pipeline import Pipeline

X, y = data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mdo', MDO()),
    ('knn', KNN())
])

pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)

print(classification_report(y_test, y_hat))






#########################################

df_train = pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")


target_count = df_train.VAR_CLASS.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Class 2:', target_count[2])
print('Proportion:0_1', round(target_count[0] / target_count[1], 2), ': 1')
print('Proportion:0_2', round(target_count[0] / target_count[2], 2), ': 1')
print('Proportion:1_2', round(target_count[1] / target_count[2], 2), ': 1')
target_count.plot(kind='bar', title='Count (target)');
#STEP2
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

labels = df_train.columns[2:]

X = df_train[labels]
y = df_train['VAR_CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
##NON RUNNA PIU'
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#################################################
#DOCUMENTAZIONE MULTICLASS DA PYTHON
# Authors: Vlad Niculae, Mathieu Blondel
# License: BSD 3 clause
"""
=========================
Multilabel classification
=========================

This example simulates a multi-label document classification problem. The
dataset is generated randomly based on the following process:

    - pick the number of labels: n ~ Poisson(n_labels)
    - n times, choose a class c: c ~ Multinomial(theta)
    - pick the document length: k ~ Poisson(length)
    - k times, choose a word: w ~ Multinomial(theta_c)

In the above process, rejection sampling is used to make sure that n is more
than 2, and that the document length is never zero. Likewise, we reject classes
which have already been chosen.  The documents that are assigned to both
classes are plotted surrounded by two colored circles.

The classification is performed by projecting to the first two principal
components found by PCA and CCA for visualisation purposes, followed by using
the :class:`sklearn.multiclass.OneVsRestClassifier` metaclassifier using two
SVCs with linear kernels to learn a discriminative model for each class.
Note that PCA is used to perform an unsupervised dimensionality reduction,
while CCA is used to perform a supervised one.

Note: in the plot, "unlabeled samples" does not mean that we don't know the
labels (as in semi-supervised learning) but that the samples simply do *not*
have a label.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


plt.figure(figsize=(8, 6))

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=True,
                                      random_state=1)

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()


########################

import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
df_train = pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")
labels = df_train.columns[2:]

X = df_train[labels]
y = df_train['VAR_CLASS']


clf = OneVsRestClassifier(SVC()).fit(X, y)
clf.predict

print(df_train)

training_test=df_train.groupby(['VAR_CLASS']).size().reset_index(name='counts')

import matplotlib.pyplot as plt
##VAR_CLASS= training_test(['VAR_CLASS'])
##COUNT= training_test(['counts'])

##plt.plot(VAR_CLASS,COUNT)

training_test.plot(kind='bar',x='VAR_CLASS',y='counts')
#################################
#LINK: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#examples-using-sklearn-multiclass-onevsrestclassifier



################################
#Ho ripreso la preparazione fatta da kevin e fatta la classif OVR
#Ancora non capisco dove vada riportato under o over 


training =pd.read_csv(r"C:\Users\casul\OneDrive\Desktop\università\DS LAB\progetto\training.csv",";")    
# verifica valori null all'interno del training
training = training.dropna()

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
    training = pd.read_csv('training.csv', sep=';')    
    #da inserire il TS
    X = training[['USAGE','KIT_ID','AVG_SPEED_DW','NUM_CLI']]
    y = training['VAR_CLASS']
    
    X = X.to_numpy()
    y = y.to_numpy()
    return (X,y)

X,y = prepareTraining()
############## sono ripartito da qua 
#https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

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
accuracy_score(y, y_pred) #0.366 





