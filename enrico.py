'''importing libraries...'''
import pandas as pd
import datetime as dt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix

test = pd.read_csv("fastweb_test.csv", sep =";")

test.tail(100)

test.groupby('KIT_ID')["NUM_CLI"].sum()


#test.ix[test['KIT_ID'] == 1512491].plot()





#test.ix[test['KIT_ID'] == 1512491,'NUM_CLI'].plot()

#test.ix[test['KIT_ID'] == 1512491,'USAGE'].plot()

#test.ix[test['KIT_ID'] == 1512491,'AVG_SPEED_DW'].plot()


#test.groupby('TS')['AVG_SPEED_DW'].mean().plot()

#test.groupby('TS')['USAGE'].sum().plot()

training = pd.read_csv("fastweb_training.csv", sep =";")

#training[training["VAR_CLASS"] == 0].groupby('KIT_ID')['VAR_CLASS'].count()

#pd.to_datetime(test['TS'])


training['TS'] = pd.to_datetime(training['TS'])

#training_gr = training.groupby('TS')['USAGE','AVG_SPEED_DW','NUM_CLI','VAR_CLASS'].sum()


#autocorrelation_plot(training_gr['AVG_SPEED_DW'])

training.head()

training = training.dropna()

'''X and y extraction'''
X = training[(['USAGE','AVG_SPEED_DW','NUM_CLI'])]
y = training[(['VAR_CLASS'])]
X = X.to_numpy()
y = y.to_numpy()
X.shape
y.shape
X.shape[0] != y.shape[0]
X1 = X.groupby('TS')['KIT_ID'].count()
X = pd.merge(X,X1,on = "TS")

'''preprocessing'''
# example of random undersampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss

X, y = make_classification(n_samples=25000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.333333,0.333333,0.333334],n_classes= 3, flip_y=0, random_state=100)
undersample = NearMiss(version=1, n_neighbors_ver3=3)
X_over, y_over = undersample.fit_resample(X,y)

for label, _ in counter.items():
	row_ix = where(y_train_over == label)[0]
	pyplot.scatter(X_train_over[row_ix, 0],
                  X_train[row_ix, 1],
                  label=str(label))
pyplot.legend()
pyplot.show()


# ENCODING
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(sparse=False)
labelencoder_y = LabelEncoder()
y_label = labelencoder_y.fit_transform(y_over).reshape(-1, 1)
y_ohc = onehotencoder.fit_transform(y_label)
y_ohc = y_ohc[:,0:2]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_over,
                                                    y_ohc,
                                                    test_size=0.3,
                                                    random_state=100)


len(X_train) == len(y_train)
len(X_test) == len(y_test)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

# Reshape
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test= y_test.reshape(-1, 1)

'''ARIMA'''
#model = ARIMA(training_gr['AVG_SPEED_DW'], order=(5,1,0))
#model_fit = model.fit(disp=0)
#print(model_fit.summary())

#residuals = pd.DataFrame(model_fit.resid)
#residuals.plot()
#pyplot.show()
#residuals.plot(kind='kde')
#pyplot.show()
#print(residuals.describe())


'''#Random_forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0,
                              tol=1e-5))

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)'''

'''from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test)'''


# Classification report, CM
from sklearn.metrics import multilabel_confusion_matrix
cm = multilabel_confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,recall_score
 
'''One Vs One'''
#NUSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
clf = make_pipeline(StandardScaler(), NuSVC())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

# SVC
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Gaussian
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,
                                random_state=0)
gpc.fit(X_train, y_train)
y_pred = gpc.predict_proba(X_test)
accuracy_score(y_test, y_pred)

'''One Vs Rest'''
# Gradien boosting
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

# LinearSVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

# LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

# SGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(),
                     SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

# Perceptron
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

# PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test).reshape(-1, 1)
accuracy_score(y_test, y_pred)

'''Fuzzy Logic Classifier'''


'''Cross Validation'''
'''# Undersampling per cross_val
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
# summarize class distribution
print(Counter(y))
# define dataset
X, y = make_classification(n_samples=30000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.333333,0.333333,0.333334],n_classes= 3, flip_y=0, random_state=100)
undersample = NearMiss(version=1, n_neighbors_ver3=3)
X_over, y_over = undersample.fit_resample(X,y)
# summarize class distribution
print(Counter(y_over))'''

# Cross Val Score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
scores = cross_val_score(clf,
                         KIT_X_over,
                         KIT_y_over,
                         cv=5,
                         scoring='accuracy')
scores.mean()




'''KIT_ID selection'''
KIT_ID = training[(training["KIT_ID"] == 1629361016) |
                  (training["KIT_ID"] == 2487219358) |
                  (training["KIT_ID"] == 3409364152)]

KIT_X = KIT_ID[(['TS','USAGE','NUM_CLI'])]
KIT_y = KIT_ID[(['VAR_CLASS'])]
KIT_X = KIT_X.to_numpy()
KIT_y = KIT_y.to_numpy()

# Undersampling KIT_ID selection
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss

KIT_X, KIT_y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.333333,0.333333,0.333334],n_classes= 3, flip_y=0, random_state=100)
undersample = NearMiss(version=1, n_neighbors_ver3=3)
KIT_X_over, KIT_y_over = undersample.fit_resample(KIT_X, KIT_y)

'''#Encoding KIT_ID selection
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(sparse=False)
labelencoder_y = LabelEncoder()
y_label = labelencoder_y.fit_transform(KIT_y_over).reshape(-1, 1)
y_ohc = onehotencoder.fit_transform(y_label)
y_ohc = y_ohc[:,0:2]'''

#Train-test split KIT_ID selection
X_train, X_test, y_train, y_test = train_test_split(KIT_X_over,
                                                    KIT_y_over,
                                                    test_size=0.3,
                                                    random_state=100)

'''# Reshape KIT_ID selcetion
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test= y_test.reshape(-1, 1)'''