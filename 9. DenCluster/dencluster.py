import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import datasets

#%% Load data

data1 = datasets.make_moons(5000, noise = 0.07)

#%% Process data

X = np.array(data1[0])

#%% Execute

ys = []
samplenums = np.arange(3, 10, 1)
epsnums = np.arange(0.01, 0.3, 0.1)
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
plt.tight_layout()
plt.savefig('Data1_DBSCAN.png')
plt.show()
#%% Load data 2

data2 = datasets.make_blobs(5000, 2, centers = 6)

#%% Process data 2

X = np.array(data2[0])

#%% Execute

ys = []
samplenums = np.arange(3, 10, 1)
epsnums = np.arange(0.01, 0.3, 0.1)
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
plt.tight_layout()
plt.savefig('Data2_DBSCAN.png')
plt.show()