import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import datasets

#%% Load data 2

'''
mitbih_test.csv - ECG Heartbeat Categorization Dataset
'''
data2 = pd.read_csv('../8. SepCluster/mitbih_test.csv', header = None)

#%% Process data 2

X = data2
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data

fig, ax = plt.subplots(1, figsize = (10, 10))
ax.scatter(X[:, 0], X[:, 1], c = None, cmap = plt.cm.Set1, edgecolor = 'k')
plt.title('Original dataset')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
#plt.savefig('Data2.png')
plt.show()

#%% Execute

ys = []
samplenums = np.arange(4, 17, 2)
epsnums = np.arange(0.1, 0.21, 0.05)
plt.figure(figsize = (20, 20))
ax = []

for samplenum in samplenums:
    for epsnum in epsnums:
        i = np.where(samplenums == samplenum)[0][0]
        j = np.where(epsnums == epsnum)[0][0]
        clf = DBSCAN(eps = epsnum, min_samples = samplenum)
        ys.append(clf.fit_predict(X))
        ax.append(plt.subplot2grid((7, 3), (i, j), colspan = 1))
        ax[-1].scatter(X[:, 0], X[:, 1], c = ys[-1], cmap = plt.cm.Set1, edgecolor = 'k')
        ax[-1].set_title('For ' + str('%.2f' % samplenum) + ' minpts, ' + str('%.2f' % epsnum) + ' eps')
        ax[-1].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
#plt.savefig('Data2_DBSCAN.png')
plt.show()

#%% Load data 1

'''
mobiletest.csv - Mobile Price Classification
'''
data1 = pd.read_csv('../8. SepCluster/mobiletest.csv')
target1 = data1['price_range']
del data1['price_range']

#%% Process data 1

X = data1
y = target1
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data

fig, ax = plt.subplots(1, figsize = (10, 10))
ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.title('Original dataset')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
#plt.savefig('Data1.png')
plt.show()

#%% Execute

ys = []
samplenums = np.arange(4, 17, 2)
epsnums = np.arange(60, 121, 30)
plt.figure(figsize = (20, 20))
ax = []

for samplenum in samplenums:
    for epsnum in epsnums:
        i = np.where(samplenums == samplenum)[0][0]
        j = np.where(epsnums == epsnum)[0][0]
        clf = DBSCAN(eps = epsnum, min_samples = samplenum)
        ys.append(clf.fit_predict(X))
        ax.append(plt.subplot2grid((7, 3), (i, j), colspan = 1))
        ax[-1].scatter(X[:, 0], X[:, 1], c = ys[-1], cmap = plt.cm.Set1, edgecolor = 'k')
        ax[-1].set_title('For ' + str('%.2f' % samplenum) + ' minpts, ' + str('%.2f' % epsnum) + ' eps')
        ax[-1].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
plt.savefig('Data1_DBSCAN.png')
plt.show()
