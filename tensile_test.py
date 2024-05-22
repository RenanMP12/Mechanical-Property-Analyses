# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:03:57 2024

@author: Renan Miranda Portela
"""

''' Import libraries '''
import os 
import csv
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

''' Open folder with data '''
address = r'C:\Users\rmportel\OneDrive - University of Waterloo\2024 - Winter Semester\Tensile Test - Transverse - Data\Binder_IT_02'
dir_list = os.listdir(address)

''' Entering the measurements '''
width = np.array([2.5])
thickness = np.array([19.8])

''' Checking folder list '''
if 'Graphics' in dir_list:
    dir_list.remove('Graphics')

if 'Spreadsheet' in dir_list:            
    dir_list.remove('Spreadsheet')
    
''' Creating a folder to store data '''
sfile = 'Spreadsheet'
if os.path.exists(address + '/' + sfile) == False:
    os.makedirs(address + '/' + sfile)
f = open(address + '/' + 'Spreadsheet' + '/' + "file.txt", "w")

''' Creating new figure '''
fig, ax = plt.subplots(figsize = (20,10))
ax.set_ylim(-50,80)
ax.set_xlim(0,0.008)
ax.legend(fontsize="30")
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.ylabel(ylabel = '\u03C3 [MPa]', fontsize=30)
plt.xlabel(xlabel = '\u03B5 [ ]',fontsize=30)
plt.grid()

for i in enumerate(dir_list):
    ''' Reading data from csv file '''
    table = pd.read_csv(address + "/"  + i[1])

    ''' Extracting strain from table and converting force to stress '''
    strain = table['ΔL/L0 [1]'].to_numpy()
    table['Stress [MPa]'] = table['Load_[N]']/(width.mean()*thickness.mean())
    stress = table['Stress [MPa]'].to_numpy()
    
    ''' Interpolating values '''
    strain_aux = np.arange(0, 0.008, 1e-4).reshape(-1, 1)
    stress_aux = np.interp(strain_aux, strain, stress).reshape(-1, 1)
    
    ''' Plotting data '''
    marker_list = ['^', 's', 'o', 'x', '*']
    label = r'Sample ' + str(i[0] + 1)
    ax.plot(strain_aux, stress_aux, label = label, marker = marker_list[i[0]], markevery = 10, markersize = 10)
    ax.legend(fontsize = 20)

    ''' Obtaining Young's modulus taking values between 0.002 and 0.004 '''
    X = table['ΔL/L0 [1]'][(table['ΔL/L0 [1]'] < 0.004) & (table['ΔL/L0 [1]'] > 0.002)].values.reshape(-1, 1)
    Y = table['Stress [MPa]'][(table['ΔL/L0 [1]'] < 0.004) & (table['ΔL/L0 [1]'] > 0.002)].values.reshape(-1, 1)
    
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
           
    Y_pred = linear_regressor.predict(strain_aux)  # make predictions
    Y_pred = Y_pred - linear_regressor.intercept_[0]
    
   # dictionary[i[0]] = strain_aux
   # dictionary[i[1]] = Y_pred