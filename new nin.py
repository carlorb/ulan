import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import seaborn as sns


plt.ioff()

rain_gap=36 #in hours
days_before=20
min_rain= 0.5 #in mm 

rain = pd.read_csv('data\\ninw.csv', usecols=[0,1], parse_dates=0, index_col=0)

rain.index.name = 'timestamp'

rain = rain.resample('30Min', how='sum')
rain['event_id'] = None
rain= rain.reset_index()

cur = pd.DataFrame(None)
temp = pd.DataFrame(None)
raincur=pd.DataFrame(None)

landslides = pd.read_csv('data//ninLLtest.txt', names = ['timestamp','event'], parse_dates=[0])
last = rain.index.max()
event_count=0
end=0
rain.event_id=0
test = pd.DataFrame()
#sorting rain data into events
try:
    while end < last:
        if event_count == 0: #if first run, basis of start is beginning of dataframe
            start = rain[rain['rain']>min_rain].index[0]
        else:
            cur = rain.loc[end+1:]
            start = cur[cur['rain']>min_rain].index[0]
        
        cur = rain.loc[start:]
        end = cur[cur['rain']<=min_rain].index[0]
        
        temp = rain.loc[end+1:]
        endtemp = temp[temp['rain']>min_rain].index[0]
        
        diff = rain['timestamp'][endtemp] - rain['timestamp'][end]
        
        while diff.total_seconds()/3600 <= rain_gap:
            end=endtemp
            temp = rain.loc[end+1:]
            endtemp = temp[temp['rain']>min_rain].index[0]
            diff = rain['timestamp'][endtemp] - rain['timestamp'][end]
        
        event_count+=1
        rain.loc[start:end,'event_id']=event_count
        
        if len(rain[end:] [rain[end:].rain >0]) == 0:
            break
        
except IndexError:
    pass

group = rain.groupby('event_id')
df = pd.DataFrame(columns=['intensity','duration','start','end','landslide'], index=range(1,rain.event_id.max()) )

for s in range(1,rain.event_id.max()):
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
#rain.set_index('timestamp', inplace=True)


for c in range(len(landslides)):  
    t=landslides.timestamp[c]
    t0 = t - timedelta(days=days_before)    
    landslide_id += 1
    other.loc[t0:t, 'landslide']=landslide_id
    
    print t,'\n',other.loc[t0:t],'\n\n\n'

correlated = other[other.landslide>=1]
rain.set_index('timestamp',inplace=True)

    

rain.loc[:,'acc'] = rain.rain.cumsum()
correlated.loc[:,'rain']=correlated['intensity']*correlated['duration']


df.loc[:,'rain']=df['intensity']*df['duration']
df.loc[:,'acc']=df.rain.cumsum()

corg=correlated.groupby('landslide').get_group


for x in range(1,correlated.landslide.max()+1):
    cur=corg(x)
    
    rstart=cur.index.min()  
    
    if cur.index.max() > landslides.timestamp[x-1]:
        rend=cur.end.max()
    else:
        rend = landslides.timestamp[x-1]+timedelta (hours=3)
        
    cur_rain=rain[rstart:rend]
    cur_rain.loc[:,'acc'] = cur_rain.rain.cumsum()
    
    fig=plt.figure(figsize=(12,6))
    ax=plt.subplot()
    for s in cur_rain.event_id.unique():     
        g=cur_rain.groupby('event_id').get_group
        dt=g(s).index
        r=g(s).rain
        a=g(s).acc
        
        rain15pt=ax.scatter(dt,r, color='red',s=1)
        rain15=ax.plot(dt,r, color='red')
        
      
        if s == 0:        
            norain=ax.scatter(dt,a,s=0.5,marker='.',color='black')
#            ax.legend(['No Rain Periods'])
            
        else:
            acc=ax.plot(dt,a) 
#            ax.legend(['Rainfall Events'])
    
    r15 = mlines.Line2D([], [], color='red',marker='.',markersize=5, label='15Min Rain')
    nr = mlines.Line2D([], [], color='black',linestyle ='x',marker='o', label='No Rain Periods')
    racc=mlines.Line2D([], [], color='black', label='Accumulated Rain')
    ll = mlines.Line2D([], [], color='blue',linestyle = '--', label='Landslide')
        
    
    ax.set_xlim([rstart,rend])
    ax.set_ylim([0,a.max() + 50])
    plt.xticks(rotation=30)
    plt.axvline(landslides.timestamp[x-1],linestyle='--')
    plt.title('Accumulated Rain for Landslide Event at ' + str(landslides.timestamp[x-1]) +', Sto Nino, Talaingod')
    plt.text(rstart+timedelta(days=0.25),a.max() / 2,'Parameters: \n Rain Gap: ' + str(rain_gap) + 'hrs \n Minimum Rain: ' + str(min_rain) + 'mm \n Rainfall ' + str(days_before) + ' days before are considered', bbox=dict(facecolor='black', alpha=0.2))
    plt.legend(handles=[r15,nr,racc,ll], loc='best')
    fig.tight_layout()
    plt.savefig('final//nin landslide ' + str(x) + '.png')
    plt.close()
    
fig=plt.figure(figsize=(12,6))
ax=plt.subplot()

x=df.duration
y=df.intensity
rains = ax.scatter(x,y, marker='o', s=10,color='black')

xl=[]
yl=[]

correlated['binary']=0
gg=correlated.groupby('landslide').get_group
for v in range(1,correlated.landslide.max()+1):
    #choose rainfall event with highest rain output as the "culprit rainfall"
    a=gg(v).sort_values(by='rain', ascending=False,).iloc[0]
    xtemp=a.duration
    ytemp=a.intensity
    xl.append(xtemp)
    yl.append(ytemp)
    lrain=ax.scatter(xtemp, ytemp, marker='*',color='red',s=100)     



ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Duration (hrs)')
ax.set_ylabel('Intensity (mm/hr)')

corr = mlines.Line2D([], [], color='red',linestyle = '..',marker='o', label='Rainfall Attributed to a Landslide')
plt.legend(handles=[corr], loc=0)
plt.text(0.12,0.12,'Parameters: \n Rain Gap: ' + str(rain_gap) + 'hrs \n Minimum Rain: ' + str(min_rain) + 'mm \n Rainfall ' + str(days_before) + ' days before are considered', bbox=dict(facecolor='black', alpha=0.2))
plt.title('Rainfall Intensity-Duration Graph of Noah Raingauge #858')
plt.savefig('final//nin ID.png')
plt.close()

  
####
# loop in line 180 same as in line 123
# optimize code
  
  
#fig, axes = plt.subplots(nrows=1,ncols=1)
#ax=rain.plot(ax=axes,x=rain.index,y='acc')#, kind='scatter',c='landslide')
#ax=plt.plot_date(dfgg.end,dfgg.acc)
#plt.show()