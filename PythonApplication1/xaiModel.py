from pathlib import Path
import os
import math
import re

import pandas as pd
import numpy as np

import data_preprocessing as dpp
import bearingFeatures as zst2_bf


def dir_to_df(experimentsFolder, vibrationSensor = False, torqueSensor = False, forceSensor = False, laserSensor = False):  

    files = os.listdir(experimentsFolder)

    def extract_number(filename):
        parts = filename.split('_')
        num = parts[1]
        match = re.search(r'\d+', num)
        return int(match.group()) if match else None

    # Find the filename with the highest number
    recent_folder = max(files, key=extract_number)

    recent_folder_path = Path(experimentsFolder, recent_folder)

    # change the Boolean values to True for required sensors
    df = dpp.recentRawDataFrame(recent_folder_path, vibrationSensor = vibrationSensor, torqueSensor = torqueSensor, forceSensor = forceSensor, laserSensor = laserSensor)

    return df

def failFeatures_freq(df):
    stats_speed1 = df['speed1'].describe().tolist()
    stats_speed2 = df['speed2'].describe().tolist()

    df_speed1 = np.abs(stats_speed1[1]) #mean
    df_speed2 = np.abs(stats_speed2[1]) #mean

    #Bearing Frequencies
    #https://webtools3.skf.com/engcalc/CalcBearingFrequencies.do
    #https://power-mi.com/content/rolling-element-bearing-components-and-failing-frequencies


    # freq_outer = (n/2)*(RPM/60)*(1-(Bd/Pd)*cos_th)  => Frequency of over-rolling point on outer ring
    # freq_inner = (n/2)*(RPM/60)*(1+(Bd/Pd)*cos_th)  => Frequency of over-rolling point on inner ring
    # freq_rotating = 2 * (1/2) * (Pd/Bd)*(RPM/60)*(1-((Bd/Pd)*(Bd/Pd)*cos_th*cos_th))  => Frequency of over-rolling rolling element 
    # freq_cage = (1/2)*(RPM/60)*(1+(Bd/Pd)*cos_th)   => Rotating frequency rolling element set & cage

    n = 9
    Bd = 12.3
    Pd = 60
    RPM1 = df_speed1
    RPM2 = df_speed2
    theta = 0

    theta_rad= theta*math.pi/180

    df_bearingFreq_RPM1 = pd.DataFrame()

    def calc_bearing_frequencies(RPM):
        freq_outer = (n/2)*(RPM/60)*(1-(Bd/Pd)*math.cos(theta_rad)) #BPFO => Frequency of over-rolling point on outer ring
        freq_inner = (n/2)*(RPM/60)*(1+(Bd/Pd)*math.cos(theta_rad)) #BPFI => Frequency of over-rolling point on inner ring
        freq_rotating = 2 * (1/2) * (Pd/Bd)*(RPM/60)*(1-((Bd/Pd)*(Bd/Pd)*math.cos(theta_rad)*math.cos(theta_rad))) #BSF => Frequency of over-rolling rolling element 
        freq_cage = (1/2)*(RPM/60)*(1-(Bd/Pd)*math.cos(theta_rad)) #FTF => Rotating frequency rolling element set & cage
        freq_innerRing = RPM/60 # => Rotating frequency inner ring
        freq_rolling = 20.756   # => Rotating frequency rolling element about its axis
        
        return (freq_outer,freq_inner, freq_rotating, freq_cage, freq_innerRing, freq_rolling)

    df_bearingFreq_RPM1 = calc_bearing_frequencies(RPM1)
    df_bearingFreq_RPM2 = calc_bearing_frequencies(RPM2)
    print("\n ---------------------------------------------------------------------")
    print('bearingFrequencies RPM1:', df_bearingFreq_RPM1)
    print('bearingFrequencies RPM2:', df_bearingFreq_RPM2)
    print("---------------------------------------------------------------------")

    i = 0
    lim = 200000
    df_force = df.actual_contactForce.values

    np.seterr(divide = 'ignore') 
    df_features = pd.DataFrame()
    df_analyze = df_force
    lim = 200000
    print("########-------- BEARING FREQUENCIES HAVE BEEN CALCULATED --------########")
    print("---------------------------------------------------------------------\n")
            
    for i in range(0,df.shape[0],lim):
        iteration = i // lim + 1
        print(f'Calculating Features: Processing {iteration}/{len(range(0, df.shape[0], lim))}')

        features = zst2_bf.calculate_features(df_analyze[i:i+lim], calc_bearing_frequencies(np.abs(np.mean(df['speed1'][i:i+lim]))), 533)
        df_features = pd.concat([df_features, features], axis=0)
        
    df_features = df_features.reset_index(drop=True)

    #Frequency Val to fail_freq_*

    # df_features[df_features.filter(like='freq').columns.tolist()] #Display only columns having freq i.e freq_1, freq_2, etc.,

    df_features_flattened = df_features.values.flatten()
    df_features_flattened = df_features_flattened[~np.isnan(df_features_flattened)]
    most_repetitive_freq = pd.Series(df_features_flattened).value_counts().index.tolist()
    fail_freq = most_repetitive_freq[:10]
    print("---------------------------------------------------------------------")
    print("########-------- FEATURES HAVE BEEN CALCULATED --------######## \n")
    print("---------------------------------------------------------------------")
            
    # Drop all other float columns except the specified ones
    df_features_all = df_features[[col for col in df_features.columns if not isinstance(col, float) or col in fail_freq]]

    df_features_all_copy = df_features_all.copy()
    
    for i in range(len(fail_freq)):
        df_features_all_copy.rename(columns={fail_freq[i]: f'fail_freq_{i+1}'}, inplace=True)

    print("########-------- FAILURE FREQUENCIES --------########")
    print('<======================================>')
    print(fail_freq)
    print('<======================================>')

    return df_features_all_copy, fail_freq
