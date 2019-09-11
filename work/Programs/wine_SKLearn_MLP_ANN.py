import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

wine = pd.read_csv('D:/Sukkur IBA 2019/Spring 2019/Machine Learning/PPTs/Lec 10/wine.csv')
wine.head(5)
wine.describe().transpose()
wine.shape

X = wine.drop('Wine',axis=1)
y = wine['Wine']
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#mlp = MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,10,10),max_iter=500, random_state=1)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

print ((mlp.coefs_[0])) # tell the weights of layer 0 and 1
print ((mlp.intercepts_[0])) # tell the bias values of layer 0 and 1
