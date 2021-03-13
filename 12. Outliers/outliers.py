import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

#%% Load data

data1 = pd.read_csv('../3. Bayes/winequality-red.csv')
del data1['quality']

data2 = pd.read_csv('banknotes.csv').iloc[:110]
del data2['conterfeit']

#%% Process data

X = data1
pca = PCA(n_components = 1)
X = pca.fit_transform(X)

#%% Plot original data

plt.figure(figsize = (20, 2))
plt.plot(X, np.zeros_like(X) + 0, 'x')
plt.show()

#%% Execute

clf = EmpiricalCovariance()
clf.fit(X)