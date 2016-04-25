import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from scipy import stats


rain_gap=24 #in hours
days_before=10

rain = pd.read_csv('data\\blcw.csv', usecols=[0,5])

rain['timestamp'] = pd.to_datetime(rain['timestamp'], format='%Y-%m-%d %H:%M:%S')
rain = rain.set_index(pd.DatetimeIndex(rain['timestamp']))
rain = rain.drop('timestamp', axis = 1)
rain.index.name = 'timestamp'

#rain = rain.resample('30Min', how='sum')
#rain = rain [rain.rain.values >= 0]
rain['event_id'] = None
rain= rain.reset_index()

cur = pd.DataFrame(None)
temp = pd.DataFrame(None)
master = pd.DataFrame(None)
raincur=pd.DataFrame(None)

landslides = pd.read_csv('data//blclandslides.txt', names = ['timestamp','event'], parse_dates=[0])
last = rain.index.max()
event_count=0
end=0

try:
    while end < last:
        if event_count == 0: #if first run, basis of start is beginning of dataframe
            start = rain[rain['rain']!=0].index[0]
        else:
            cur = rain.loc[end+1:]
            start = cur[cur['rain']!=0].index[0]
        
        cur = rain.loc[start:]
        end = cur[cur['rain']==0].index[0]
        
        temp = rain.loc[end+1:]
        endtemp = temp[temp['rain']!=0].index[0]
        
        diff = rain['timestamp'][endtemp] - rain['timestamp'][end]
        
        while diff.total_seconds()/3600 <= rain_gap:
            end=endtemp
            temp = rain.loc[end+1:]
            endtemp = temp[temp['rain']!=0].index[0]
            diff = rain['timestamp'][endtemp] - rain['timestamp'][end]
            
        cur = rain[start:end+1]
        event_count +=1
        cur.event_id = event_count 
        master=master.append(cur)
        
except IndexError:
    pass

group = master.groupby('event_id')
df = pd.DataFrame(columns=['intensity','duration','start','end','landslide'], index=range(1,master.event_id.max()) )

for s in range(1,master.event_id.max()):
    g = group.get_group
    start = g(s).timestamp.min() 
    end = g(s).timestamp.max() 
    D =  end - start 
    D = D.total_seconds() / 3600
    I = g(s).rain.sum() / D
    df.loc[s] = pd.Series({'intensity':I, 'duration':D, 'start':start, 'end':end,})


#df is dataframe indexed by 'id' containing  I, D, start,end, landslide,rain,acc
df.index.name = 'id'
df.landslide=0

#other is just 'df' sorted accordning to start date
other=df.copy()
other['id']=other.index
other.set_index('start',inplace=True)

landslide_id=0
master.set_index('timestamp', inplace=True)


for c in range(len(landslides)):  
    t=landslides.timestamp[c]
    t0 = t - timedelta(days=days_before)
#    x=other.loc[t0:t]
#    i = other.index.searchsorted(t)
#    x=other.ix[other.index[i]].id
    
#    x['landslide'] = 1
    test=other.loc[t0:t]
    
    landslide_id += 1
    other[test.id.values[0]:test.id.values[-1]]['landslide'] = landslide_id
    
    print t,'\n',other.loc[t0:t],'\n\n\n'

correlated = other[other.landslide>=1]
cumcorrelated = test
rain.set_index('timestamp',inplace=True)




#for x in correlated.iterrows():
#    rstart=x[1][2]
#    rend=x[1][3]
#    raincur = raincur.append(master[rstart:rend])
    

rain['acc'] = rain.rain.cumsum()
correlated['rain']=correlated['intensity']* correlated['duration']

#fig, axes = plt.subplots(nrows=1,ncols=1)
#ax=df.plot(ax=axes,x='duration',y='intensity',c='landslide',cmap='gnuplot',kind='scatter',logx=True,logy=True)
#plt.show()

df['rain']=df['intensity']*df['duration']
df['acc']=df.rain.cumsum()

dfg=df.groupby('landslide').get_group
dfgg = dfg(1)

fig, axes = plt.subplots(nrows=1,ncols=1)
ax=rain.plot(ax=axes,x=rain.index,y='acc')#, kind='scatter',c='landslide')
ax=plt.plot_date(dfgg.end,dfgg.acc)
plt.show()