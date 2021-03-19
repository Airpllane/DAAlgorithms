import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from pyod.models.hbos import HBOS

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

#%% Load data

'''
banknotes.csv - Swiss banknote counterfeit detection
'''
data1 = pd.read_csv('banknotes.csv').iloc[:110]
del data1['conterfeit']

'''
winequality-red.csv - Red Wine Quality
'''
data2 = pd.read_csv('../3. Bayes/winequality-red.csv')
del data2['quality']

#%% Process data 1

X = data1
pca = PCA(n_components = 1)
X = pca.fit_transform(X)

#%% Plot original data 1

plt.figure(figsize = (12, 2))
plt.plot(X, np.zeros_like(X) + 0, 'x')
plt.yticks([])
plt.title('Banknotes')
plt.xlabel('PCA feature')
plt.savefig('Original1.png')
plt.show()

#%% Execute, plot 1

clf = EmpiricalCovariance()
clf.fit(X)
clfc = clf.score(X)
clf2 = HBOS()
clf2.fit(X)
clf2c = clf2.decision_function(X).mean()

fig, ax = plt.subplots(2, figsize = (12, 4))
for x in X:
    ax[0].plot(x, 0, 'bx' if clf.score(x) > clfc else 'rx')
    ax[0].set_title('Max likelihood')
    ax[0].set(xlabel = 'PCA feature')
    ax[0].set_yticks([])
    ax[1].plot(x, 0, 'bx' if clf2.decision_function(np.reshape(x, (-1, 1))) < clf2c else 'rx')
    ax[1].set_title('Histogram-based')
    ax[1].set(xlabel = 'PCA feature')
    ax[1].set_yticks([])
plt.tight_layout()
plt.savefig('Anomaly1.png')
plt.show()

#%% Process data 2

X = data2
pca = PCA(n_components = 1)
X = pca.fit_transform(X)

#%% Plot original data 2

plt.figure(figsize = (12, 2))
plt.plot(X, np.zeros_like(X) + 0, 'x')
plt.yticks([])
plt.title('Red wine quality')
plt.xlabel('PCA feature')
plt.savefig('Original2.png')
plt.show()

#%% Execute, plot 2

clf = EmpiricalCovariance()
clf.fit(X)
clfc = clf.score(X)
clf2 = HBOS()
clf2.fit(X)
clf2c = clf2.decision_function(X).mean()

fig, ax = plt.subplots(2, figsize = (12, 4))
for x in X:
    ax[0].plot(x, 0, 'bx' if clf.score(x) > clfc else 'rx')
    ax[0].set_title('Max likelihood')
    ax[0].set(xlabel = 'PCA feature')
    ax[0].set_yticks([])
    ax[1].plot(x, 0, 'bx' if clf2.decision_function(np.reshape(x, (-1, 1))) < clf2c else 'rx')
    ax[1].set_title('Histogram-based')
    ax[1].set(xlabel = 'PCA feature')
    ax[1].set_yticks([])
plt.tight_layout()
plt.savefig('Anomaly2.png')
plt.show()