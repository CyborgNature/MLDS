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
