# 3+12 rainfall series
import pandas as pd
from datetime import timedelta as delt
import matplotlib.pyplot as plt

#l is the dataframe containing landslide details
#r is the dataframe containing rain details
l = pd.read_csv('data//ninLLtest.txt', names = ['timestamp','event'], parse_dates=[0],index_col=1)
r = pd.read_csv('data//ninw.csv', usecols=[0,1], parse_dates=[0], index_col=0)

r = r.resample('30T', how='sum')

r['r12'] = pd.rolling_sum(r.rain,576,min_periods=1)
r['r3'] = pd.rolling_sum(r.rain,144,min_periods=1)

x = []
y = []
for m in range(1,len(l)+1):
    cur=l.loc[m] 
    x.append(r['r3'][cur])
    y.append( r['r12'][cur-delt(days=3)])
    
plt.scatter(x,y)

