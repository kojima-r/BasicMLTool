import os
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data = []
label = []
for line in open("iris.data"):
    arr = line.strip().split(",")
    if len(arr) > 1:
        data.append(arr[:4])
        label.append(arr[4])

le.fit(label)
y = le.transform(label)
fp = open("dataset.csv", "w")
for i, d in enumerate(data):
    s = ",".join(d) + "," + str(y[i])
    fp.write(s + "\n")
