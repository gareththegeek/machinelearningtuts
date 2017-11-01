import numpy as np
import pandas as pd

data_file = ".\\data\\ad\\ad.data"
data = pd.read_csv(data_file, sep=",", header=None, low_memory=False)


def toNum(cell):
    try:
        return np.float(cell)
    except:
        return np.nan


def seriesToNum(series):
    return series.apply(toNum)


train_data = data.iloc[0:, 0:-1].apply(seriesToNum)
train_data = train_data.dropna()


def toLabel(string):
    if string == "ad.":
        return 1
    else:
        return 0


train_labels = data.iloc[train_data.index, -1].apply(toLabel)

from sklearn.svm import LinearSVC

clf = LinearSVC()
clf.fit(train_data[100:2300], train_labels[100:2300])

def classify(row):
    p = clf.predict(row.reshape(1, -1))
    if p == 1:
        print("Ad")
    else:
        print("Not Ad")

classify(train_data.iloc[12])
classify(train_data.iloc[-1])
