import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

#%% Load data

'''
unempl.csv - US Unemployment Rate by County, 1990-2016
'''
data = pd.read_csv('unempl.csv')[['Year', 'Month', 'Rate']].groupby(['Year', 'Month']).sum()

#%%

data.plot()