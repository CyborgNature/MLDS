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

    
