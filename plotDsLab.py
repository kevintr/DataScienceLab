# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:41:32 2020

@author: TranchinaKe
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as pyplot


##Lettura dataset iniziale 
#training = pd.read_csv('training.csv', sep=';')  
## verifica valori null all'interno del training
#training = training.dropna()
#training['TS'] = pd.to_datetime(training['TS'])


#funzione di normalizzazione necessaria per sovrappore i grafici
def normalizeSeries(seriesInDatframe):
    if(seriesInDatframe.max()-seriesInDatframe.min() ==0):
        seriesInDatframe=0
    else:
        seriesInDatframe = (seriesInDatframe-seriesInDatframe.min())/(
        seriesInDatframe.max()-seriesInDatframe.min())
    return seriesInDatframe

#Il metodo normalizza Usage NumCli e VarClass, prima di costituire il grafico per ognuno dei 3 valori e poi sovrapporli
def plotUsageAndNumcliAndVarClassByTS(dataFrame,pred):
    if(pred == False):
        #Normalize 
        dataFrame.loc[:,'USAGE']= normalizeSeries(dataFrame.loc[:,'USAGE'])
        dataFrame.loc[:,'VAR_CLASS']= normalizeSeries(dataFrame.loc[:,'VAR_CLASS'])
        dataFrame.loc[:,'NUM_CLI']= normalizeSeries(dataFrame.loc[:,'NUM_CLI'])
        
        pyplot.figure(figsize=(15,2))
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'USAGE'],linewidth=1)
        pyplot.scatter(dataFrame.loc[:,'TS'],dataFrame.loc[:,'VAR_CLASS'], color='red',linewidth=None,edgecolors=None , marker='o')
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'NUM_CLI'], color='c',linewidth=3)
        pyplot.xticks(np.arange(min(dataFrame['TS']), max(dataFrame['TS'])+timedelta(days=1), timedelta(days=1)),rotation=70)
        pyplot.legend(('USAGE', 'NUM_CLI', 'VAR_CLASS'))
        pyplot.show()
    else:
        #Normalize 
        dataFrame.loc[:,'USAGE']= normalizeSeries(dataFrame.loc[:,'USAGE'])
        dataFrame.loc[:,'VAR_CLASS']= normalizeSeries(dataFrame.loc[:,'VAR_CLASS'])
        dataFrame.loc[:,'VAR_CLASS_PRED']= normalizeSeries(dataFrame.loc[:,'VAR_CLASS_PRED'])
        dataFrame.loc[:,'NUM_CLI']= normalizeSeries(dataFrame.loc[:,'NUM_CLI'])
        
        pyplot.figure(figsize=(15,2))
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'USAGE'],linewidth=1)
        pyplot.scatter(dataFrame.loc[:,'TS'],dataFrame.loc[:,'VAR_CLASS'], color='red',linewidth=0.25,edgecolors=None , marker='o')
        pyplot.scatter(dataFrame.loc[:,'TS'],dataFrame.loc[:,'VAR_CLASS_PRED'] + 0.25, color='green',linewidth=0.25,edgecolors=None , marker='o')
        pyplot.plot(dataFrame.loc[:,'TS'],dataFrame.loc[:,'NUM_CLI'], color='c',linewidth=3)
        pyplot.xticks(np.arange(min(dataFrame['TS']), max(dataFrame['TS'])+timedelta(days=1), timedelta(days=1)),rotation=70)
        pyplot.legend((
                'USAGE', 
                       'NUM_CLI', 
                       'VAR_CLASS',
                       'VAR_CLASS_PRED'))
        pyplot.show()
    return pyplot.show()
#
#training2 = training[training['VAR_CLASS'] == 2]
#training2['KIT_ID'].unique()# trovare gli unici KIT_ID che hanno avuto un disservizio di tipo 1
#
#training1 = training[training['VAR_CLASS'] == 1]
#training1['KIT_ID'].unique()# trovare gli unici KIT_ID che hanno avuto un disservizio di tipo 1
#
#kit3409364152 = training.loc[(training.loc[:,'KIT_ID'] == 3409364152)]
#kit1629361016 = training.loc[(training.loc[:,'KIT_ID'] == 1629361016)]
#kit2487219358 = training.loc[(training.loc[:,'KIT_ID'] == 2487219358)]
#
#print("3409364152"+str(kit3409364152.loc[:,'AVG_SPEED_DW'].unique()))
#print("1629361016"+str(kit1629361016.loc[:,'AVG_SPEED_DW'].unique()))
#print("2487219358"+str(kit2487219358.loc[:,'AVG_SPEED_DW'].unique()))
#
#plotUsageAndNumcliAndVarClassByTS(kit3409364152)
#plotUsageAndNumcliAndVarClassByTS(kit1629361016)
#plotUsageAndNumcliAndVarClassByTS(kit2487219358)
#
#provakit1_test['TS'].unique()
#
######################Prove per mettere null ai valori di ts che non sono presenti #########
#giorno8TS = pd.Series(kit3409364152[kit3409364152['TS'].dt.day == 8]['TS'].to_numpy())
#giorno8Usage = pd.Series(kit3409364152[kit3409364152['TS'].dt.day == 8]['USAGE'].to_numpy())
#giorno8NUM_Cli = pd.Series(kit3409364152[kit3409364152['TS'].dt.day == 8]['NUM_CLI'].to_numpy())
#giorno8varclass = pd.Series(kit3409364152[kit3409364152['TS'].dt.day == 8]['VAR_CLASS'].to_numpy())
#giorno8 = pd.Series(giorno8)
#new_row = {'TS':giorno8,'USAGE':giorno8Usage,'VAR_CLASS':giorno8varclass,'NUM_CLI':giorno8NUM_Cli}
##append row to the dataframe
#kit1629361016 = kit1629361016.append(new_row, ignore_index=True)
#kit1629361016.loc[:,'TS'] = pd.datetime(kit1629361016.loc[:,'TS'])
#kit1629361016.loc[:,'TS'] = kit1629361016.loc[:,'TS'].append(pd.Series(giorno8))
#kit1629361016
#kit1629361016[kit1629361016['TS'].dt.day == 8]
######################Prove per mettere null ai valori di ts che non sono presenti #########
#
#    
###################  Grafici separati ######################################################
#kit2487219358 = training[training['KIT_ID'] == 2487219358]
##kit2487219358.plot(x='TS',y='AVG_SPEED_DW',color='red')#costante
#kit2487219358.plot(x='TS',y='USAGE',color='red',figsize=(15,2.5), linewidth=1, fontsize=10)
#kit2487219358.plot(x='TS',y='NUM_CLI',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
#kit2487219358.plot(x='TS',y='VAR_CLASS',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
#kit2487219358.plot(x='TS',y='AVG_SPEED_DW',color='blue',figsize=(15,2.5), linewidth=1, fontsize=10)#costante
#plt.show()
###################  Grafici separati ######################################################
#
#
#z = training[training['VAR_CLASS']==1]['KIT_ID']
#z = training[training['VAR_CLASS']==2]['KIT_ID']
#z = training[training['VAR_CLASS']==0]['KIT_ID']
#len(z.unique())
#z.unique()
#
## Pie plot KIT_ID con disservizio e senza
#labels = 'Kit senza disservizio', 'Kit con disservizio'
#sizes = [1977, 3]
#colors = [ 'green', 'red']
#explode = (0, 0)  # explode 1st slice
#
## Plot
#pyplot.pie(sizes, explode=explode, labels=labels, colors=colors,
#autopct='%1.1f%%', shadow=True, startangle=140)
#
#pyplot.axis('equal')
#pyplot.show()
#
#
#print('0 ' + str(len(training[training['VAR_CLASS'] == 0])))#16521526
#print('1 ' + str(len(training[training['VAR_CLASS'] == 1])))#36
#print('2 ' + str(len(training[training['VAR_CLASS'] == 2])))#472
#
#training['VAR_CLASS'].value_counts()
#
#type(test.loc[0,'TS'])
#test.loc[:,'TS'] = pd.to_datetime(test.loc[:,'TS'])