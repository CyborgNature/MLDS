import pandas as pd
import quandl
import math
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model 
from numpy import genfromtxt

file = genfromtxt ("iris.data" , delimiter = "," , dtype = "str")
#print (file)

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
#print (file [0])

trainingSet = file [:130]
testingSet = file [130:]
#print (trainingSet.shape)
#print (testingSet.shape)


trainingX = trainingSet [:,[0,1,2,3]]
trainingX = trainingX.astype (float)
trainingY = trainingSet [:,[4]]

testingX = testingSet [:,[0,1,2,3]]
testingX = testingX.astype (float)
testingY = testingSet [:,[4]]

lr = linear_model.LogisticRegression ()
lr.fit (trainingX , trainingY)

print ("Actual data of instance [12] = " + str(testingX [12]))
print ("Actual class of instance [12] = " + str(testingY [12]))
print ("Predicted class = " + str (lr.predict([testingX [12]])))

lr.score (testingX , testingY) * 100
