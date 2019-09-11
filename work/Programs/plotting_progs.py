import pandas as pd
import quandl
import math
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model 
from numpy import genfromtxt
import matplotlib.pyplot as plot

file = genfromtxt ("iris.data" , delimiter = "," , dtype = "str")

dic = {}
count = 0 
for val in file:
    if val[4] not in dic:
        dic [val[4]] = count
        count += 1

#print (dic)
#print (file[0])

for val in file:
    val[4]=dic[val [4]]
    
features = file [:,[0,1,2,3]]
features = features.astype (float)
targets = file [:,[4]]
targets = targets.astype (int)
featuresAll = []
for observation in features:
    featuresAll.append([observation [0] + observation [1] + observation [2] + observation [3]])
plot.scatter (featuresAll , targets , color = 'red' , alpha = 1)
plot.rcParams['figure.figsize'] = [10,8]
plot.title('Iris Dataset Scatter Plot')
plot.xlabel ('Features')
plot.ylabel ('Class')
plot.show ()

    
featuresAll = []
targets = []
for feature in features:
    featuresAll.append (feature [0])
    targets.append (feature [1])
groups = ('Iris-setosa' , 'Iris-versicolor' , 'Iris-virginica')
colors = ('red' , 'blue' , 'green')
data = ((featuresAll [:50] , targets [:50]) , (featuresAll [50:100] , targets [50:100]) , (featuresAll [100:150] , targets [100:150]))
for item,color,group in zip (data , colors , groups):
    x , y = item
    plot.scatter (x,y , color = color , alpha = 1)
plot.rcParams['figure.figsize'] = [15,12]
plot.title('Iris Dataset Scatter Plot')
plot.xlabel ('Sepal Length')
plot.ylabel ('Sepal Width')
plot.show ()


import matplotlib .pyplot as plt
import pandas as pd
df = pd.read_csv("iris.csv")

ratio = df['PetalLengthCm']/df['PetalWidthCm']
ratio1 = df['SepalLengthCm']/df['SepalWidthCm']



for name,group in df.groupby ('Species'):
    plt.scatter (group.index , ratio [group.index] , label = name)
plt.legend ()
plt.show ()

for name,group in df.groupby ('Species'):
    plt.scatter (group.index , ratio1 [group.index] , label = name)
plt.legend ()
plt.show ()
