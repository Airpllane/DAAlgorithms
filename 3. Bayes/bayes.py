import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#%%

data1 = pd.read_csv('mushrooms.csv')
target1 = data1['class']
del data1['class']

'''
data2 = pd.read_csv('winequality-red.csv')
target2 = data2['quality']
del data2['quality']
'''

#%%

v = DictVectorizer(sparse=False)
X1 = v.fit_transform(data1.to_dict('records'))
y1 = target1.replace({'p': 1, 'e': 0})

'''
X2 = data2
y2 = (target2 > 5).astype(int)
'''

#%%

X = X1
y = y1

#%%

clf = GaussianNB()
metrics = {'TrainAccuracy': [], 'TrainPrecision': [], 'TrainRecall': [], 'TrainF1': [],
           'TestAccuracy': [], 'TestPrecision': [], 'TestRecall': [], 'TestF1': []}

splits = np.arange(0.6, 0.9, 0.05)
for split in splits:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = split)
    clf.fit(X_train, y_train)
    
    metrics['TrainAccuracy'].append(accuracy_score(clf.predict(X_train), y_train))
    metrics['TestAccuracy'].append(accuracy_score(clf.predict(X_test), y_test))
    
    metrics['TrainPrecision'].append(precision_score(clf.predict(X_train), y_train))
    metrics['TestPrecision'].append(precision_score(clf.predict(X_test), y_test))
    
    metrics['TrainRecall'].append(recall_score(clf.predict(X_train), y_train))
    metrics['TestRecall'].append(recall_score(clf.predict(X_test), y_test))
    
    metrics['TrainF1'].append(f1_score(clf.predict(X_train), y_train))
    metrics['TestF1'].append(f1_score(clf.predict(X_test), y_test))

#%%

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (15, 10))
ax[0, 0].plot(splits, metrics['TrainAccuracy'])
ax[0, 0].plot(splits, metrics['TestAccuracy'])
ax[0, 0].legend(['Train set', 'Test set'])
ax[0, 0].set_title('Accuracy')
ax[0, 0].set(xlabel = 'Train set fraction', ylabel = 'Accuracy')

ax[0, 1].plot(splits, metrics['TrainPrecision'])
ax[0, 1].plot(splits, metrics['TestPrecision'])
ax[0, 1].legend(['Train set', 'Test set'])
ax[0, 1].set_title('Precision')
ax[0, 1].set(xlabel = 'Train set fraction', ylabel = 'Precision')

ax[1, 0].plot(splits, metrics['TrainRecall'])
ax[1, 0].plot(splits, metrics['TestRecall'])
ax[1, 0].legend(['Train set', 'Test set'])
ax[1, 0].set_title('Recall')
ax[1, 0].set(xlabel = 'Train set fraction', ylabel = 'Recall')

ax[1, 1].plot(splits, metrics['TrainF1'])
ax[1, 1].plot(splits, metrics['TestF1'])
ax[1, 1].legend(['Train set', 'Test set'])
ax[1, 1].set_title('F1')
ax[1, 1].set(xlabel = 'Train set fraction', ylabel = 'F1')

#plt.savefig('metrics.png')
plt.show()