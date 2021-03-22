import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matrixprofile as mp
import matplotlib.ticker as ticker

#%% Load data

'''
unempl.csv - US Unemployment Rate by County, 1990-2016
'''
data = pd.read_csv('unempl.csv')[['Year', 'Month', 'Rate']]#.groupby(['Year', 'Month']).mean()
data['Time'] = pd.to_datetime(data['Year'].astype(str) + ' ' + data['Month'])
del data['Year']
del data['Month']
data = data.groupby(['Time']).mean()

#%% To array

X = np.array(data)

#%% Plot original data

dates = pd.date_range(start = '19900101', end = '20170101', freq = 'm').strftime('%Y-%m').to_list()
plt.figure(figsize = (16, 6))
plt.plot(dates, X, 'g')
plt.title('Unemployment by year')
plt.xlabel('Month')
plt.ylabel('Unemployment rate, %')
plt.xticks(rotation = 90)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(6))
#plt.savefig('OriginalTime.png')
plt.show()

#%% Execute, plot

wsize = 24
k = 2

profile = mp.compute(X.squeeze(), wsize)
profile = mp.discover.discords(profile, exclusion_zone = wsize, k = k)
mp.visualize(profile)
#plt.savefig('DiscordTime.png')
plt.show()

mp_adjusted = np.append(profile['mp'], np.zeros(profile['w'] - 1) + np.nan)
fig, ax = plt.subplots(1, 1, sharex = True, figsize = (16, 6))
ax.plot(np.arange(len(profile['data']['ts'])), profile['data']['ts'])
ax.set_title('Highlighted anomaly')
ax.set_xlabel('Time')
ax.set_ylabel('Unemployment')
ax.set_xticks([])
ax.set_yticks([])
flag = 1
for discord in profile['discords']:
    x = np.arange(discord, discord + profile['w'])
    y = profile['data']['ts'][discord:discord + profile['w']]
    if flag:
        ax.plot(x, y, c = 'r',label = "Discord")
        flag = 0
    else:
        ax.plot(x, y, c = 'r')
plt.legend()
#plt.savefig('AnomalyTime.png')
plt.show()