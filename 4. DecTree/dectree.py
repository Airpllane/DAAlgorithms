import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#%% Load data
'''
data = pd.read_csv('../3. Bayes/mushrooms.csv')
target = data['class']
del data['class']
'''

data = pd.read_csv('../3. Bayes/winequality-red.csv')
target = data['quality']
del data['quality']

#%% Process data
'''
v = DictVectorizer(sparse=False)
X = v.fit_transform(data.to_dict('records'))
y = target.replace({'p': 1, 'e': 0})
'''

X = data
y = (target > 6).astype(int)

#%% Execute, plot trees

clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
metrics = {'TrainAccuracy': [], 'TrainPrecision': [], 'TrainRecall': [], 'TrainF1': [],
           'TestAccuracy': [], 'TestPrecision': [], 'TestRecall': [], 'TestF1': []}

splits = np.arange(0.6, 0.95, 0.1)
fig, ax = plt.subplots(4, figsize = (40, 50))
for split in splits:
    i = np.where(splits == split)[0][0]
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
    
    #plot_tree(clf, filled = True, feature_names = v.get_feature_names(), ax = ax[i])
    plot_tree(clf, filled = True, feature_names = X.columns, ax = ax[i])

#plt.savefig('trees.png')
plt.show()    

#%% Metrics

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

#%% Save metrics

with open('metrics_4.json', 'w') as fp:
    #json.dump(metrics, fp, indent = 4)
    pass