import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

#%% Load data

'''
banknotes.csv - Swiss banknote counterfeit detection
'''
data1 = pd.read_csv('../12. Outliers/banknotes.csv').iloc[:110]
del data1['conterfeit']

'''
winequality-red.csv - Red Wine Quality
'''
data2 = pd.read_csv('../3. Bayes/winequality-red.csv')
del data2['quality']

#%% Process data 1

X = data1
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data 1

plt.figure(figsize = (8, 8))
plt.scatter(X[:, 0], X[:, 1])
plt.title('Banknotes')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
#plt.savefig('Original1.png')
plt.show()

#%% Execute, plot 1

fig, ax = plt.subplots(2, figsize = (8, 6))

y = [0 for i in X]
for i in X:
    il = np.where(X == i)[0][0]
    count = 0
    for j in X:
        if np.linalg.norm(i) != np.linalg.norm(j) and np.linalg.norm(i - j) <= 0.6:
            count += 1
        if count >= np.ceil(0.03 * X.shape[0]):
            y[il] = 1
            break
ax[0].scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
ax[0].set_title('Nested loop')
ax[0].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')

clf = DBSCAN(eps = 0.7, min_samples = 10)
y = clf.fit_predict(X)


ax[1].scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
ax[1].set_title('Clustering')
ax[1].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
#plt.savefig('Anomaly1.png')
plt.show()

#%% Process data 2

data2 = data2.drop(np.random.choice(data2.index, 1200, replace=False))
X = data2
pca = PCA(n_components = 2)
X = pca.fit_transform(X)

#%% Plot original data 2

plt.figure(figsize = (8, 8))
plt.scatter(X[:, 0], X[:, 1])
plt.title('Red wine')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
#plt.savefig('Original2.png')
plt.show()

#%% Execute, plot 2

fig, ax = plt.subplots(2, figsize = (8, 6))

y = [0 for i in X]
for i in X:
    il = np.where(X == i)[0][0]
    count = 0
    for j in X:
        jl = np.where(X == j)[0][0]
        if il != jl and np.linalg.norm(i - j) <= 40:
            count += 1
            if count >= np.ceil(0.1 * X.shape[0]):
                y[il] = 1
                break
ax[0].scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
ax[0].set_title('Nested loop')
ax[0].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')

clf = DBSCAN(eps = 35, min_samples = 80)
y = clf.fit_predict(X)


ax[1].scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
ax[1].set_title('Clustering')
ax[1].set(xlabel = 'PCA feature 1', ylabel = 'PCA feature 2')
plt.tight_layout()
#plt.savefig('Anomaly2.png')
plt.show()
