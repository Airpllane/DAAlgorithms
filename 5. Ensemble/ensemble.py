import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import BaggingClassifier
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

#%% Execute

metrics = {'TrainAccuracy': [], 'TrainPrecision': [], 'TrainRecall': [], 'TrainF1': [],
           'TestAccuracy': [], 'TestPrecision': [], 'TestRecall': [], 'TestF1': []}

est_nums = np.arange(50, 110, 10)
for est_num in est_nums:
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = BaggingClassifier(n_estimators = est_num)
    clf.fit(X_train, y_train)
    
    metrics['TrainAccuracy'].append(accuracy_score(clf.predict(X_train), y_train))
    metrics['TestAccuracy'].append(accuracy_score(clf.predict(X_test), y_test))
    
    metrics['TrainPrecision'].append(precision_score(clf.predict(X_train), y_train))
    metrics['TestPrecision'].append(precision_score(clf.predict(X_test), y_test))
    
    metrics['TrainRecall'].append(recall_score(clf.predict(X_train), y_train))
    metrics['TestRecall'].append(recall_score(clf.predict(X_test), y_test))
    
    metrics['TrainF1'].append(f1_score(clf.predict(X_train), y_train))
    metrics['TestF1'].append(f1_score(clf.predict(X_test), y_test))

#%% Load prev. metrics

with open('../3. Bayes/metrics_3.json', 'r') as fp:
    metrics_3 = json.load(fp)

with open('../4. Dectree/metrics_4.json', 'r') as fp:
    metrics_4 = json.load(fp)

min_points = min(len(metrics['TestAccuracy']), len(metrics_3['TestAccuracy']), len(metrics_4['TestAccuracy']))

#%% Metrics

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (15, 10))
ax[0, 0].plot(est_nums, metrics['TrainAccuracy'])
ax[0, 0].plot(est_nums, metrics['TestAccuracy'])
ax[0, 0].plot(est_nums[:len(metrics_3['TrainAccuracy'])], metrics_3['TrainAccuracy'][:min(len(est_nums), len(metrics_3['TrainAccuracy']))])
ax[0, 0].plot(est_nums[:len(metrics_3['TestAccuracy'])], metrics_3['TestAccuracy'][:min(len(est_nums), len(metrics_3['TestAccuracy']))])
ax[0, 0].plot(est_nums[:len(metrics_4['TrainAccuracy'])], metrics_4['TrainAccuracy'][:min(len(est_nums), len(metrics_4['TrainAccuracy']))])
ax[0, 0].plot(est_nums[:len(metrics_4['TestAccuracy'])], metrics_4['TestAccuracy'][:min(len(est_nums), len(metrics_4['TestAccuracy']))])
ax[0, 0].plot(est_nums[:min_points], metrics_4['TestAccuracy'][:min_points])
ax[0, 0].legend(['Train set', 'Test set', 'Bayes train set', 'Bayes test set', 'DecTree train set', 'DecTree test set'])
ax[0, 0].set_title('Accuracy')
ax[0, 0].set(xlabel = 'Number of estimators (mixed)', ylabel = 'Accuracy')

ax[0, 1].plot(est_nums, metrics['TrainPrecision'])
ax[0, 1].plot(est_nums, metrics['TestPrecision'])
ax[0, 1].plot(est_nums[:len(metrics_3['TrainPrecision'])], metrics_3['TrainPrecision'][:min(len(est_nums), len(metrics_3['TrainPrecision']))])
ax[0, 1].plot(est_nums[:len(metrics_3['TestPrecision'])], metrics_3['TestPrecision'][:min(len(est_nums), len(metrics_3['TestPrecision']))])
ax[0, 1].plot(est_nums[:len(metrics_4['TrainPrecision'])], metrics_4['TrainPrecision'][:min(len(est_nums), len(metrics_4['TrainPrecision']))])
ax[0, 1].plot(est_nums[:len(metrics_4['TestPrecision'])], metrics_4['TestPrecision'][:min(len(est_nums), len(metrics_4['TestPrecision']))])
ax[0, 1].legend(['Train set', 'Test set', 'Bayes train set', 'Bayes test set', 'DecTree train set', 'DecTree test set'])
ax[0, 1].set_title('Precision')
ax[0, 1].set(xlabel = 'Number of estimators (mixed)', ylabel = 'Precision')

ax[1, 0].plot(est_nums, metrics['TrainRecall'])
ax[1, 0].plot(est_nums, metrics['TestRecall'])
ax[1, 0].plot(est_nums[:len(metrics_3['TrainRecall'])], metrics_3['TrainRecall'][:min(len(est_nums), len(metrics_3['TrainRecall']))])
ax[1, 0].plot(est_nums[:len(metrics_3['TestRecall'])], metrics_3['TestRecall'][:min(len(est_nums), len(metrics_3['TestRecall']))])
ax[1, 0].plot(est_nums[:len(metrics_4['TrainRecall'])], metrics_4['TrainRecall'][:min(len(est_nums), len(metrics_4['TrainRecall']))])
ax[1, 0].plot(est_nums[:len(metrics_4['TestRecall'])], metrics_4['TestRecall'][:min(len(est_nums), len(metrics_4['TestRecall']))])
ax[1, 0].legend(['Train set', 'Test set', 'Bayes train set', 'Bayes test set', 'DecTree train set', 'DecTree test set'])
ax[1, 0].set_title('Recall')
ax[1, 0].set(xlabel = 'Number of estimators (mixed)', ylabel = 'Recall')

ax[1, 1].plot(est_nums, metrics['TrainF1'])
ax[1, 1].plot(est_nums, metrics['TestF1'])
ax[1, 1].plot(est_nums[:len(metrics_3['TrainF1'])], metrics_3['TrainF1'][:min(len(est_nums), len(metrics_3['TrainF1']))])
ax[1, 1].plot(est_nums[:len(metrics_3['TestF1'])], metrics_3['TestF1'][:min(len(est_nums), len(metrics_3['TestF1']))])
ax[1, 1].plot(est_nums[:len(metrics_4['TrainF1'])], metrics_4['TrainF1'][:min(len(est_nums), len(metrics_4['TrainF1']))])
ax[1, 1].plot(est_nums[:len(metrics_4['TestF1'])], metrics_4['TestF1'][:min(len(est_nums), len(metrics_4['TestF1']))])
ax[1, 1].legend(['Train set', 'Test set', 'Bayes train set', 'Bayes test set', 'DecTree train set', 'DecTree test set'])
ax[1, 1].set_title('F1')
ax[1, 1].set(xlabel = 'Number of estimators (mixed)', ylabel = 'F1')

#plt.savefig('metrics.png')
plt.show()