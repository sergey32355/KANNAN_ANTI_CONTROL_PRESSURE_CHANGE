import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
import time
from pathlib import Path
import numpy as np

def datCol(dataFile, low=False, abstandWS=False, prev=False, speedCol=False, triggerCol=False):
    data = pd.read_csv(dataFile, sep="\t", encoding='unicode_escape', header=None, low_memory=False)
    data.drop(data.index[[0,1]], inplace=True)
    # data.drop(data.columns[[2]], axis=1, inplace=True)
    # data.columns = ['phase', 'time', 'distance', 'vibration', 'torque2', 'real_force', 'actual_contactForce', 'actual_speed1',
    #                      'actual_speed2', 'laserSensor_wear', 'torque1', 'drive_power1', 'drive_power2', 'trigger']

    if(low):
        if(abstandWS):
            data.drop(data.columns[[2,3,6,12,13, 14]], axis=1, inplace=True)
            data.columns = ['phase','time','vibration','torque2','actual_contactForce','speed1','speed2','laserSensor_wear','torque1']
        else:
            data.drop(data.columns[[2,5,11,12,13,14,15]], axis=1, inplace=True)
            data.columns = ['phase','time','vibration','torque2','actual_contactForce','speed1','speed2','laserSensor_wear','torque1']
    elif(prev):
        data.columns = ['time','vibration','torque1','torque2','actual_contactForce','laserSensor_wear']
    elif(speedCol):
        data.columns = ['time','vibration','torque1','torque2','actual_contactForce','laserSensor_wear','speed1','speed2']
    elif(triggerCol):
        data.columns = ['time','vibration','torque1','torque2','actual_contactForce','laserSensor_wear','speed1','speed2', 'trigger_key']
    
    #Replace "," with "."
    data[data.columns] = data[data.columns].replace(',', '.', regex=True)

    #Changing the data types
    if(low):
        ignore = ['phase']
    else:
        ignore = []
    
    data = (data.set_index(ignore, append=True).astype(float).reset_index(ignore))
    #data = data.apply(pd.to_numeric, downcast='float')
    
    return data

def datColIndividual(dataFile, vibrationSensor = False, torqueSensor = False, forceSensor = False, laserSensor = False):
    data = pd.read_csv(dataFile, sep="\t", encoding='unicode_escape', low_memory=False)
    data.drop(data.index[[0,1]], inplace=True)
    
    if(vibrationSensor):
        # df_indivual = data.iloc[:, [0, 4, 6, 7]] #Force
        df_indivual = data[['Timestamp', 'Schwingungsueberwachung', 'Istdrehzahl-1', 'Istdrehzahl-2']]
        df_indivual.columns = ['time','vibration', 'speed1','speed2']
    elif(torqueSensor):
        df_indivual = data[['Timestamp', 'Drehmoment-1', 'Drehmoment-2', 'Istdrehzahl-1', 'Istdrehzahl-2']]
        df_indivual.columns = ['time', 'torque1', 'torque2', 'speed1','speed2']
    elif(forceSensor):
        df_indivual = data[['Timestamp', 'Anpresskraft', 'Istdrehzahl-1', 'Istdrehzahl-2']]
        df_indivual.columns = ['time','actual_contactForce', 'speed1','speed2']
    elif(laserSensor):
        df_indivual = data[['Timestamp', 'Lasersensor-Verschleiss', 'Istdrehzahl-1', 'Istdrehzahl-2']]
        df_indivual.columns = ['time','laserSensor_wear', 'speed1','speed2']

    #Replace "," with "."
    # df_indivual[df_indivual.columns] = df_indivual[df_indivual.columns].replace(',', '.', regex=True)
    
    df_indivual_copy = df_indivual.copy()
    df_indivual_copy[df_indivual_copy.columns] = df_indivual_copy[df_indivual_copy.columns].replace(',', '.', regex=True)
    df_indivual = df_indivual_copy
    
    #Changing the data types
    ignore = []
    
    df_indivual = (df_indivual.set_index(ignore, append=True).astype(float).reset_index(ignore))

    return df_indivual

def datDataframe(dataFile, experimentNum, data_frequency, schwingungOne=False, vibrationSensor = False, torqueSensor = False, forceSensor = False, laserSensor = False):
    if((data_frequency == 10) | schwingungOne):
        if(experimentNum < 107):
            #print(str(experimentNum) + ": 10 Hz Frequency: Experiments before 107")
            data = datCol(dataFile, low=True)
            data_df = data[data['phase'].str.contains('Messung')]
        elif(schwingungOne & (107 <= experimentNum < 143)):
            #print(str(experimentNum) + ": 1 KHz Frequency: Experiments between 107 and 142 (Single File)")
            data = datCol(dataFile, prev=True)
            data_df = data
        elif(schwingungOne & (143 <= experimentNum < 313)):
            #print(str(experimentNum) + ": 1 KHz Frequency: Experiments after 143 (Single File)")
            data = datCol(dataFile, speedCol=True)
            data_df = data
        elif(schwingungOne & (experimentNum >= 313)):
            #print(str(experimentNum) + ": 1 KHz Frequency: Experiments after 313 (Single File)")
            data = datCol(dataFile, triggerCol=True)
            data_df = data
        else:
            #print(str(experimentNum) + ": 10 Hz Frequency: Experiments after 106")
            data = datCol(dataFile, low=True, abstandWS = True)
            data_df = data[data['phase'].str.contains('Messung')]
    
    elif(vibrationSensor | torqueSensor | forceSensor | laserSensor):
        data_df = datColIndividual(dataFile, vibrationSensor, torqueSensor, forceSensor, laserSensor)

    else:
        if(experimentNum < 143):
            #print(str(experimentNum) + ": 1 KHz Frequency: Experiments before 143")
            data_schwingung = pd.DataFrame()
            for i in range(len(dataFile)):
                data = datCol(dataFile[i], prev=True)
                data_schwingung = data_schwingung.append(data)
        elif(143 <= experimentNum < 313):
            #print(str(experimentNum) + ": 1 KHz Frequency: Experiments after 143")
            data_schwingung = pd.DataFrame()
            for i in range(len(dataFile)):
                data = datCol(dataFile[i], speedCol=True)
                data_schwingung = data_schwingung.append(data)
        elif(experimentNum >= 313):
            #print(str(experimentNum) + ": 1 KHz Frequency: Experiments after 313")
            data_schwingung = pd.DataFrame()
            for i in range(len(dataFile)):
                data = datCol(dataFile[i], triggerCol=True)
                data_schwingung = data_schwingung.append(data)
                        
            
        data_df = data_schwingung
        
    return data_df


def datPath(experimentsPath, data_frequency, schwingungOne = True):
    print(experimentsPath)
    temp_expNum = int(re.findall('\d+', str(experimentsPath))[-1])
    if(data_frequency == 10):
        schwingungOne = False
    
    dataFileSchwingung = []
    for id, searchFile in enumerate(glob.glob(str(experimentsPath)+ "/**/*.txt", recursive=True), start=1):
        if (data_frequency == 10):
            dataFile = searchFile
        elif(temp_expNum>=313):
            substring = "MesskanaeleHF"
            if re.search(substring, searchFile):
                dataFile = searchFile
                dataFileSchwingung.append(dataFile)
                if(schwingungOne):
                    dataFile = dataFileSchwingung[0]
                else:
                    dataFile = dataFileSchwingung  
        else:
            substring = "Schwingung"
            if re.search(substring, searchFile):
                dataFile = searchFile
                dataFileSchwingung.append(dataFile)
                
                if(schwingungOne):
                    dataFile = dataFileSchwingung[0]
                else:
                    dataFile = dataFileSchwingung  
            
    return dataFile

def datRawDataFrame(experimentsFolder, expNumStart, expNumEnd, vibrationSensor = False, torqueSensor = False, forceSensor = False, laserSensor = False):
    expFile = os.listdir(experimentsFolder)
    experimentsPath = []
    for i in range(len(expFile)):
        expNum = re.findall('\d+', expFile[i])[1]
        if( expNumStart <= int(expNum) <= expNumEnd):
            expPath = Path(experimentsFolder, 'zst2_' + expNum)
            experimentsPath.append(expPath)

    start_1 = time.time()
    data_frequency = 10000
    schwingungOne = False
    schwingungLoop = True

    if(data_frequency == 10):
        schwingungOne = False
        schwingungLoop = False

    experimentGroupsPath = []
    for j in range(len(experimentsPath)):
        dataFile = datPath(experimentsPath[j], data_frequency, schwingungOne)
        sortedDataFile = sorted(dataFile, key=len)
        if(data_frequency == 10):
            experimentGroupsPath = np.append(experimentGroupsPath, dataFile)
        else:
            experimentGroupsPath = np.append(experimentGroupsPath, sortedDataFile)

    df = pd.DataFrame()
    temp_expTime = 0
    
    end_1 = time.time()
    print('Data sorting time (s): ' + str(end_1 - start_1))

    start = time.time()

    for k in range(len(experimentGroupsPath)):
        currentExp = experimentGroupsPath[k].split('\\')[-2]
        currentExpNum = re.findall('\d+', currentExp)[1]
        start_2 = time.time()
        if(schwingungLoop):
            schwingungOne = True
            
            data_df = datDataframe(experimentGroupsPath[k], int(currentExpNum), data_frequency, schwingungOne = False, vibrationSensor = vibrationSensor, torqueSensor = torqueSensor, forceSensor = forceSensor, laserSensor = laserSensor)
            
            if(int(currentExpNum) >= 313):
                perc = k%4
                if perc == 0:
                    title = currentExp + '_' + str(int(k/4)) + 'h15m'
                elif perc == 1:
                    title = currentExp + '_' + str(int(k/4)) + 'h30m'
                elif perc == 2:
                    title = currentExp + '_' + str(int(k/4)) + 'h45m'
                elif perc == 3:
                    title = currentExp + '_' + str(int(k/4)+1) + 'h'
                
            else:
                perc = k%2
                if perc == 0:
                    title =  currentExp + '_' + str(int(k/2)) + 'h30m'
                else:
                    title = currentExp + '_' + str(int(k/2)+1) + 'h'
            #Method
            data_df['expNum'] = currentExpNum
            # mergedData = [df, data_df]
            # df = pd.concat(mergedData, ignore_index=True)


            if(len(df)==0):
                data_df['expTime'] = data_df['time']/3600 
                df = data_df
            elif (df.iloc[-1].expNum !=data_df.iloc[0].expNum):
                temp_expTime = df.iloc[-1].expTime
                data_df['expTime'] = data_df['time']/3600 + temp_expTime
                mergedData = [df, data_df]
                df = pd.concat(mergedData, ignore_index=True)
            else:
                data_df['expTime'] = data_df['time']/3600 + temp_expTime
                mergedData = [df, data_df]
                df = pd.concat(mergedData, ignore_index=True)
            
            end_2 = time.time()
            print('Experiment: ' + str(title) + '\t | Processing time (s): ' + str(end_2 - start_2))

    print("\n ---------------------------------------------------------------------")
    print(df.head())
    return df



def recentRawDataFrame(recentExperimentsFolder, vibrationSensor = False, torqueSensor = False, forceSensor = False, laserSensor = False):
    
    start_1 = time.time()
    data_frequency = 10000
    schwingungOne = False
    schwingungLoop = True

    experimentGroupsPath = []
    
    dataFile = datPath(recentExperimentsFolder, data_frequency, schwingungOne)
    sortedDataFile = sorted(dataFile, key=len)
    if(data_frequency == 10):
        experimentGroupsPath = np.append(experimentGroupsPath, dataFile)
    else:
        experimentGroupsPath = np.append(experimentGroupsPath, sortedDataFile)

    df = pd.DataFrame()
    temp_expTime = 0
    
    end_1 = time.time()
    print('Data sorting time (s): ' + str(end_1 - start_1))
    print("---------------------------------------------------------------------")
    print("########-------- PROCESSING THE DATA --------######## ")
    print("---------------------------------------------------------------------")
    
    
    start = time.time()

    for k in range(len(experimentGroupsPath)):
        
        currentExp = experimentGroupsPath[k].split('\\')[-2]
        currentExpNum = re.findall('\d+', currentExp)[1]
        start_2 = time.time()
        if(schwingungLoop):
            schwingungOne = True
            data_df = datDataframe(experimentGroupsPath[k], int(currentExpNum), data_frequency, schwingungOne = False, vibrationSensor = vibrationSensor, torqueSensor = torqueSensor, forceSensor = forceSensor, laserSensor = laserSensor)
            if(int(currentExpNum) >= 313):
                perc = k%4
                if perc == 0:
                    title = currentExp + '_' + str(int(k/4)) + 'h15m'
                elif perc == 1:
                    title = currentExp + '_' + str(int(k/4)) + 'h30m'
                elif perc == 2:
                    title = currentExp + '_' + str(int(k/4)) + 'h45m'
                elif perc == 3:
                    title = currentExp + '_' + str(int(k/4)+1) + 'h'
                
            else:
                perc = k%2
                if perc == 0:
                    title =  currentExp + '_' + str(int(k/2)) + 'h30m'
                else:
                    title = currentExp + '_' + str(int(k/2)+1) + 'h'
            #Method
            data_df['expNum'] = currentExpNum
            # mergedData = [df, data_df]
            # df = pd.concat(mergedData, ignore_index=True)
           
            if(len(df)==0):
                data_df['expTime'] = data_df['time']/3600 
                df = data_df
            elif (df.iloc[-1].expNum !=data_df.iloc[0].expNum):
                temp_expTime = df.iloc[-1].expTime
                data_df['expTime'] = data_df['time']/3600 + temp_expTime
                mergedData = [df, data_df]
                df = pd.concat(mergedData, ignore_index=True)
            else:
                data_df['expTime'] = data_df['time']/3600 + temp_expTime
                mergedData = [df, data_df]
                df = pd.concat(mergedData, ignore_index=True)
            
            end_2 = time.time()
            # print('Experiment: ' + str(title) + '\t | Processing time (s): ' + str(end_2 - start_2) + ' | ' + str(k+1) +'/'+ str(len(experimentGroupsPath)))
            print('Processing Experiment: ' + str(title) + '\t | ' + str(k+1) +'/'+ str(len(experimentGroupsPath)))

    print("\n ---------------------------------------------------------------------")
    print("########-------- DATA HAS BEEN PREPROCESSED --------######## ")
    print("---------------------------------------------------------------------")
    
    print(df.head())
    
    return df