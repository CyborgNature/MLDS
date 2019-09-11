import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import numpy as np
df = pd.read_csv("weather.csv")
df.head()
df.replace('?', -99999, inplace = True)
from sklearn.preprocessing import LabelEncoder
wea = LabelEncoder()
df['outlook'] = wea.fit_transform(df['outlook'].values)
df['temperature'] = wea.fit_transform(df['temperature'].values)
df['humidity'] = wea.fit_transform(df['humidity'].values)
df['windy'] = wea.fit_transform(df['windy'].values)
x = np.array(df.drop(['play'], 1))
y = np.array(df['play'])
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x,y, test_size = 0.2)
clf = GaussianNB()
clf.fit(X_train, Y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
result = clf.score(X_test, Y_test)
arr = np.array([[1, 2, 0, 1]])
check = clf.predict(arr)
print(check)
