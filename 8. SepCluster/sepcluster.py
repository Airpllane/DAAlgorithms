import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn import datasets

#%% Load data

'''
s1.csv - S1
'''
#data = pd.read_csv('s1.csv')
data1 = datasets.make_blobs(5000, 2, centers = 6)

#%% Process data

X = np.array(data1[0])

#%% Execute

ys = []
clusternums = np.arange(3, 10, 1)
#fig, ax = plt.subplots(4, figsize = (40, 50))
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
plt.tight_layout()
plt.savefig('Orig_KMeans.png')
plt.show()

#%% Add noise

for row in X:
    if(np.random.randint(11) < 3):
        col = np.random.randint(2)
        row[col] += (np.random.randint(-1, 2) * 0.1 * row[col]) 

#%% Execute (KMeans)

ys = []
clusternums = np.arange(3, 10, 1)
#fig, ax = plt.subplots(4, figsize = (40, 50))
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
plt.tight_layout()
plt.savefig('Noise_KMeans.png')
plt.show()

#%% Execute (KMedoids)

ys = []
clusternums = np.arange(3, 10, 1)
#fig, ax = plt.subplots(4, figsize = (40, 50))
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMedoids(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
plt.tight_layout()
plt.savefig('Noise_KMedoids.png')
plt.show()

#%% Load data 2

data2 = datasets.make_moons(5000, noise = 0.1)

#%% Process data 2

X = np.array(data2[0])

#%% Execute (KMeans)

ys = []
clusternums = np.arange(3, 10, 1)
#fig, ax = plt.subplots(4, figsize = (40, 50))
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMeans(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
plt.tight_layout()
plt.savefig('Data2_KMeans.png')
plt.show()

#%% Execute (KMedoids)

ys = []
clusternums = np.arange(3, 10, 1)
#fig, ax = plt.subplots(4, figsize = (40, 50))
plt.figure(figsize = (20, 20))
ax = []
for clusternum in clusternums:
    i = np.where(clusternums == clusternum)[0][0]
    clf = KMedoids(n_clusters = clusternum)
    ys.append(clf.fit_predict(X))
    ax.append(plt.subplot2grid((4, 2), (int(i / 2), int(i % 2)), colspan = (1 if i < 6 else 2)))
    ax[i].scatter(X[:, 0], X[:, 1], c = ys[i], cmap = plt.cm.Set1, edgecolor = 'k')
plt.tight_layout()
plt.savefig('Data2_KMedoids.png')
plt.show()

#%%
'''
fig = plt.figure(1, figsize = (8, 6))
ax = Axes3D(fig)
#ax = ax = fig.add_subplot(111, projection = '3d')
ax.view_init(30, 30)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = None, cmap = plt.cm.Set1, edgecolor = 'k', s = 20)

ax.xaxis._axinfo['juggled'] = (0,0,0)
ax.yaxis._axinfo['juggled'] = (1,1,1)
ax.zaxis._axinfo['juggled'] = (2,2,2)
'''