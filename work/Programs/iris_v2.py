import pandas as pd
import quandl
import math
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("iris.csv")
df.head (1)

print (df.shape)
print(df.groupby('Species').size())
print (df.describe())

df = (df.drop(['Id'], 1))

df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

df.hist()
plt.show()

scatter_matrix(df)
plt.show()

array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

lr = linear_model.LogisticRegression ()
lr.fit (X_train , Y_train)

predictions = lr.predict (X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
