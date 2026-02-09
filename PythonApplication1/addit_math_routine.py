#https://github.com/Minoru938/KmdPlus/blob/main/tutorial.ipynb

# Import libraries.
import pandas as pd
import numpy as np
# from pymatgen.core.composition import Composition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys
import threading
import random
import time
import winsound
from datetime import datetime
import pandas as pd
import os
import casadi as cs
import control as ct
from pathlib import Path
from numpy.fft import fft, ifft, rfft
from scipy.signal import savgol_filter
from scipy.signal import stft
import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels

import PySide6
from   PySide6 import QtUiTools,QtWidgets
from   PySide6.QtGui import *
from   PySide6.QtWidgets import QApplication,QMainWindow,QVBoxLayout,QWidget,QLabel
from   PySide6 import QtCore
from   PySide6.QtWidgets import QFileDialog

import matplotlib
from   matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from   matplotlib.figure import Figure
import matplotlib.pyplot as plt
from   matplotlib.colors import Normalize, LogNorm, NoNorm

#******************************************************************************************************
#******************************************************************************************************
#******************************MOTOR ARM MODEL*********************************************************
#******************************************************************************************************
#******************************************************************************************************

#virtual models for working with control

# model of a spring loaded arm driven by a motor:
#https://python-control.readthedocs.io/en/latest/nonlinear.html#nonlinear-system-models

class Motor_Arm:

    def __init__(self,t_interval=10):
        self.servomech_params = {
                                    'J': 100,             # Moment of inertia of the motor
                                    'b': 10,              # Angular damping of the arm
                                    'k': 1,               # Spring constant
                                    'r': 1,               # Location of spring contact on arm
                                    'l': 2,               # Distance to the read head
                                }
        
        self.servomech=None
        self.InitModel()
        #all current variables
        self.t_interval=t_interval #timeinterval during which we observe the system response to our input
        self.Reset()

    # State derivative
    def servomech_update(self, t, x, u, params):
        # Extract the configuration and velocity variables from the state vector
        
        theta = x[0]                # Angular position of the disk drive arm
        thetadot = x[1]             # Angular velocity of the disk drive arm
        tau = u[0]                  # Torque applied at the base of the arm
    
        # Get the parameter values
        J, b, k, r = map(params.get, ['J', 'b', 'k', 'r'])
    
        # Compute the angular acceleration
        dthetadot = 1/J * (
            -b * thetadot - k * r * np.sin(theta) + tau)
    
        # Return the state update law
        return np.array([thetadot, dthetadot])

    # System output (tip radial position + angular velocity)
    def servomech_output(self,t, x, u, params):
        l = params['l']
        return np.array([l * x[0], x[1]])

    def InitModel(self):
        # System dynamics
        self.servomech = ct.nlsys(self.servomech_update, 
                                  self.servomech_output, 
                                  name='servomech',
                                  params=self.servomech_params, 
                                  states=['theta', 'thdot'],
                                  outputs=['y', 'thdot'], 
                                  inputs=['tau'])      
    def Reset(self):        
        self.cur_state=[1.0,0.2]
        self.cur_output=[0,0]
        self.cur_input=[0]
        self.t=0      

    def Run(self,U):
        timepts = np.linspace(self.t, self.t+self.t_interval)
        U1 = np.ones((len(timepts)))*U
        resp1 = ct.input_output_response(self.servomech, timepts, U1,self.cur_state)
        time=resp1.time
        output=resp1.outputs[0]
        
        self.t=self.t+self.t_interval

        self.cur_state=resp1.states[:,np.shape(resp1.states)[1]-1]
        self.cur_output=resp1.outputs[:,np.shape(resp1.outputs)[1]-1]#resp1.outputs
        self.cur_input= resp1.inputs[:,np.shape(resp1.inputs)[1]-1]#resp1.inputs
        #print(np.shape(self.cur_state))        
        return resp1.time, resp1.outputs[0]

"""
#example
model=Motor_Arm()
model.Reset()
time,output=model.Run(3.0)
plt.plot(time,output)
time,output=model.Run(0.02)
plt.plot(time,output)
time,output=model.Run(-10.02)
plt.plot(time,output)
"""


#*********************************************************************************************************
#*********************************************************************************************************
#*****************************************KERNEL MEAN EMBEDDINGS******************************************
#*********************************************************************************************************
#*********************************************************************************************************

class KMD():
    """
    Kernel mean descriptor (KMD).
    """
    def __init__(self, method = "1d"):
        """
        Parameters
        ----
        method: str, default = "1d"
              method must be "md" or "1d".
              For "md", KMD is generated on a multidimensional feature space.
              For "1d", KMD is generated for each feature, then combined.
        ----
        """
        self.method = method
    
    def transform(self, weight, component_features, n_grids = None, sigma = "auto", scale = True):
        """
        Generate kernel mean descriptor (KMD) with the Gaussian kernel (materials → descriptors).
        
        Args
        ----
        weight: array-like of shape (n_samples, n_components)
              Mixing ratio of constituent elements that make up each sample.
        component_features: array-like of shape (n_components, n_features)
              Features for each constituent element.
        n_grids: int, default = None
              The number of grids for discretizing the kernel mean.
              The kernel mean is discretized at the n_grids equally spaced grids 
              between a maximum and minimum values for each feature.
              This argument is only necessary for "1d".
        sigma: str or float, default = "auto"
              A hyper parameter defines the kernel width.
              If sima = "auto", the kernel width is given as the inverse median of the nearest distances
              for "md", and as the inverse of the grid width for "1d".
        scale: bool, default = True
              IF scale = True, component_features is scaled.
        Returns
        ----
        KMD: numpy array of shape (n_samples, n_components) for "md", and (n_samples, n_features*n_grids) for "1d".
        """  
        self.component_features = component_features
        self.sigma = sigma
        self.scale = scale
        # Generate KMD on a multidimensional feature space.
        if self.method == "md":
          
            # Standardize each feature to have mean 0 and variance 1 (for "md").
            if scale == True:
                component_features = (component_features - component_features.mean(axis=0))/component_features.std(axis=0, ddof=1)
            else:
                pass
            
            # Set the kernel width as the inverse median of the nearest distances.
            if sigma == "auto":
                d = distance_matrix(component_features, component_features)**2
                min_dist = [np.sort(d[i,:])[1] for i in range(component_features.shape[0])] # the nearest distances
                gamma = 1/median(min_dist)
                kernelized_component_features = np.exp(-d * gamma)
                KMD = np.dot(weight, kernelized_component_features)
                return KMD
            
            # Manually set the kernel width.
            else:
                d = distance_matrix(component_features, component_features)**2
                kernelized_component_features = np.exp(-d/(2*sigma**2))
                KMD = np.dot(weight, kernelized_component_features)
                return KMD
         
        # Generate KMD for each feature, then combine them.
        elif self.method == "1d":
            
            if n_grids == None:
                print('For self.method = "1d", please set n_grids')
                return
            else:
                pass
            
            # Min-Max Normalization (for "1d").
            if scale == True:
                component_features = (component_features - component_features.min(axis=0))/(component_features.max(axis=0) - component_features.min(axis=0))
            else:
                pass
            
            # Set the kernel width as the inverse of the grid width.
            if sigma == "auto":
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    gamma = 1/(grid_points[1] - grid_points[0])**2
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d*gamma))
                kernelized_component_features = np.concatenate(k, axis=1)
                KMD = np.dot(weight, kernelized_component_features)
                return KMD
                    
            # Manually set the kernel width.
            else:
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d/(2*sigma**2)))
                kernelized_component_features = np.concatenate(k, axis=1)
                KMD = np.dot(weight, kernelized_component_features)
                return KMD       
        else:
            print('self.method must be "md" or "1d"')
    
    def inverse_transform(self, KMD): 
        """
        Derive the weights of the constituent elements for a given kernel mean descriptors 
        by solving a quadratic programming (descriptors → materials).
        
        Args
        ----
        KMD: array-like of shape (n_samples, n_components) for "md", (n_samples, n_features*n_grids) for "1d".
              Kernel mean descriptor (KMD).
        Returns
        ----
        weight: numpy array of shape (n_samples, n_components).
        """  
        component_features = self.component_features
        sigma = self.sigma
        scale = self.scale
        if self.method == "md":
            
            # Standardize each feature to have mean 0 and variance 1 (for "md").
            if scale == True:
                component_features = (component_features - component_features.mean(axis=0))/component_features.std(axis=0, ddof=1)
            else:
                pass
            
            KMD = np.asarray(KMD)
            n_components = KMD.shape[1]
            
            # Set the kernel width as the inverse median of the nearest distances.
            if sigma == "auto":
                d = distance_matrix(component_features, component_features)**2
                min_dist = [np.sort(d[i,:])[1] for i in range(component_features.shape[0])] # the nearest distances
                gamma = 1/median(min_dist)
                kernelized_component_features = np.exp(-d * gamma)
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: smaller sigma may solve the problem")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
                    
            # Manually set the kernel width.
            else:
                d = distance_matrix(component_features, component_features)**2
                kernelized_component_features = np.exp(-d/(2*sigma**2))
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: smaller sigma may solve the problem")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
            
        elif self.method == "1d":
            
            KMD = np.asarray(KMD)
            n_grids = int(KMD.shape[1]/component_features.shape[1])
            
            # Min-Max Normalization (for "1d").
            if scale == True:
                component_features = (component_features - component_features.min(axis=0))/(component_features.max(axis=0) - component_features.min(axis=0))
            else:
                pass
            
            # Set the kernel width as the inverse of the grid width.
            if sigma == "auto":
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    gamma = 1/(grid_points[1] - grid_points[0])**2
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d*gamma))
                kernelized_component_features = np.concatenate(k, axis=1)
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: consider increasing the number of grids (n_grids)")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
            
            # Manually set the kernel width.
            else:
                max_cf = component_features.max(axis=0) 
                min_cf = component_features.min(axis=0)
                x = np.asarray(component_features)
                k = []
                for i in range(component_features.shape[1]):
                    grid_points = np.linspace(min_cf[i], max_cf[i], n_grids) 
                    d = np.array([(x[j,i] - grid_points)**2 for j in range(x.shape[0])])
                    k.append(np.exp(-d/(2*sigma**2)))
                kernelized_component_features = np.concatenate(k, axis=1)
                P = np.dot(kernelized_component_features, kernelized_component_features.T) 
                if min(np.linalg.eigvals(P)) <= 0:
                    print("Given KMD is not inversible: consider increasing the number of grids (n_grids)")
                    return
                else:
                    pass
                # Equality constraints.
                A = np.ones(P.shape[0])
                b = np.array([1.])
                # Inequality constraints.
                G = np.diag(-A)
                h = np.zeros(P.shape[0])
                # Solve quadratic programming.
                w_raw = np.array([solve_qp(P, -np.dot(kernelized_component_features, KMD[i])
                                                     , G, h, A, b, solver="quadprog") for i in range(KMD.shape[0])])
                w = np.round(abs(w_raw), 12)       
                weight = w/w.sum(axis=1)[:, None]  
                return weight
                
        else:
            print('self.method must be "md" or "1d"')
            
def StatsDescriptor(weight, component_features, stats = ["mean", "var", "max", "min"]):
    """
    Generate descriptors for mixture systems using summary statistics.

    Args
    ----
    weight: array-like of shape (n_samples, n_components)
          Mixing ratio of constituent elements that make up each sample.
    component_features: array-like of shape (n_components, n_features)
          Features for each constituent element.
    stats: a list of str, default = ["mean", "var", "max", "min"]
          Type of summary statistics for generating descriptors.
          Only "mean", "var", "max" and "min" are supported.
    Returns
    ----
    SD: numpy array of shape (n_samples, n_features*len(stats)).
    """  
    w = np.asarray(weight)
    cf = np.asarray(component_features)
    n_samples = w.shape[0]

    s = []
    for x in stats:
        # Weighted mean.
        if x == "mean":
            wm = np.dot(w, cf)
            s.append(wm)
        # Weighted variance.
        elif x == "var":
            wm = np.dot(w, cf)
            wv = np.array([np.dot(w[i], (cf - wm[i])**2) for i in range(n_samples)])
            s.append(wv)
        # Maximum pooling.
        elif x == "max":
            nonzero = (w != 0) 
            maxp = np.array([cf[nonzero[i]].max(axis = 0) for i in range(n_samples)])
            s.append(maxp)
        # Minimum pooling.
        elif x == "min":
            nonzero = (w != 0) 
            minp = np.array([cf[nonzero[i]].min(axis = 0) for i in range(n_samples)])
            s.append(minp)
        else:
            print(f'"{x}" is not supported: only "mean", "var", "max" and "min" are supported as stats')

    SD = np.concatenate(s, axis = 1)
    return SD