# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:47:16 2024

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

# file address
address = r'C:\Users\rmportel\OneDrive - University of Waterloo\2024 - Winter Semester\Short beam\Binder Act 01'
path = address
dir_list = os.listdir(path)

# checking list
if 'Graphics' in dir_list:
    dir_list.remove('Graphics')

if 'Spreadsheet' in dir_list:            
    dir_list.remove('Spreadsheet')
    
sfile = 'Spreadsheet'
if os.path.exists(address + '/' + sfile) == False:
    os.makedirs(address + '/' + sfile)
f = open(address + '/' + 'Spreadsheet' + '/' + "file.txt", "w")

dictionary = {}
min_length = 100

# plot figures
fig1, ax1 = plt.subplots()    
plt.ylabel(ylabel = '$\u03C4_{12}$ [MPa]', fontsize=15)
plt.xlabel(xlabel = 'Displacement [mm]',fontsize=15)
plt.xlim(0, 1)     # set the xlim to left, right
plt.ylim(0, 60)     # set the xlim to left, right
plt.grid()

# plot figures
fig2, ax2 = plt.subplots()    
plt.grid()

for i in enumerate(dir_list):
    print(f'---------------------- Sample {i[0] + 1:d} ----------------------')
    
    # converting file extension from txt to csv
    table_1 = convert_csv(address, i[1])
    
    # converting value from voltage signal to load and displacement
    table_1 = convert_Data(table_1, i[1], i[0], width, thickness, address)
    
    # adjusting off-set
    table_1 = Off_Set_Adjustment(table_1, i)
    
    # interpolating values
    table_2 = interpolate(address, i[1], table_1)       
    
    # plotting shear stress vs. displacement
    x = table_2['Displacement [mm]'].to_numpy()
    y = table_2['Shear Stress [MPa]'].to_numpy()
    marker_list = ['^', 's', 'o', 'x', '*']
    ax1.plot(x, y, label = i[1], marker = marker_list[i[0]], markevery = 10000, markersize = 10)
    ax1.legend()
    
    # identifying array up to max value
    dictionary = filtering_slope(table_2, i, ax2, dictionary)
    
    # saving data 
    sfile = 'Spreadsheet'
    if os.path.exists(address + '/' + sfile) == False:
        os.makedirs(address + '/' + sfile)
        
    name = i[1].rstrip(".txt")
    
    # create a excel writer object
    with pd.ExcelWriter(address + '/' + sfile + '/' + name + '.xlsx') as writer:
        # use to_excel function and specify the sheet_name and index 
        # to store the dataframe in specified sheet
        table_1.to_excel(writer, sheet_name="Original Data", index=False, engine='xlsxwriter')
        table_2.to_excel(writer, sheet_name="Interpolated Data", index=False, engine='xlsxwriter')

    
table_3 = statistical_analysis(address, dictionary)

x = table_3['Displacement [mm]'].to_numpy()
mean = table_3['Average'].to_numpy()
std = table_3['Std'].to_numpy()

print(f'\n') 
print(f'------------- Stastistical Analysis -------------') 
cv = std[1:]/mean[1:]*100
print(f'Coefficient of Variance = {cv[1]:.2f}%') 
f.write(f'Coefficient of Variance = {cv[1]:.2f}%')
f.close()

# plot average and std deviation
fig3, ax3 = plt.subplots()
ax3.plot(x, mean)
ax3.fill_between(x, mean + std, mean - std, facecolor = 'blue', alpha = 0.15)

ax3.set_ylim(0,60)
ax3.set_xlim(0,0.8)
plt.ylabel(ylabel = '$\u03C4_{12}$ [MPa]', fontsize=15)
plt.xlabel(xlabel = 'Displacement [mm]',fontsize=15)
plt.grid()

# saving plots
title = address[34:]
fig1.savefig(address + '/' + 'Graphics' + '/' + title + '_experimental_data' + '.png')
fig2.savefig(address + '/' + 'Graphics' + '/' + title + '_fitted_data' + '.png')
fig3.savefig(address + '/' + 'Graphics' + '/' + title + '_statistical_data' + '.png')
   
def convert_csv(address, name):
    # Read the text file        
    file_name = address + '/' + name 
    with open(file_name, 'r') as in_file:
        data = [line.strip().split('\t') for line in in_file]
        data = data[22:]
        
        file_name = name + '.csv'
        with open(file_name, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(data)
            
    table = pd.read_csv(file_name)
    
    return table

def convert_Data(table, name, var, width, thickness, address):
    # converting values from voltage to force and displacement
    _force_conversion = 8896.44
    _displacement_conversion = 12.165
    table['Force [N]'] = table['LVDT']*_force_conversion
    table['Shear Stress [MPa]'] = 0.75*(table['Force [N]']/(width[var]*thickness[var]))
    table['Displacement [mm]'] = table['Load Cell']*_displacement_conversion
    
    return table

def Off_Set_Adjustment(table, var):
    # set initial y-axis equal to zero
    cond = True
    n = 500
    count = 0
    vector = np.ones(len(table['Displacement [mm]'].values.reshape(-1, 1)), dtype=bool)
    
    while cond == True:
        X = table['Displacement [mm]'][count*n :(count+1)*n].values.reshape(-1, 1)    
        Y = table['Shear Stress [MPa]'][count*n :(count+1)*n].values.reshape(-1, 1)    
        
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        
        if abs(linear_regressor.coef_ ) < 20:
            vector[count*n :(count+1)*n] = 0
            count += 1
        else:
            cond = False
            
    table = table[vector]    
    table = table.reset_index(drop=True)
        
    y_off_set = table['Shear Stress [MPa]'][0]
    table['Shear Stress [MPa]'] = table['Shear Stress [MPa]'] - y_off_set
    
    # set initial x-axis equal to zero
    x_off_set = table['Displacement [mm]'][0]
    table['Displacement [mm]'] = table['Displacement [mm]'] - x_off_set
    
    stress = -min(table['Shear Stress [MPa]'])
    print(f'Max Shear Stress = {stress:.2f} MPa')
    
    return table

def interpolate(address, name, table):
    x = np.absolute(table['Displacement [mm]'].to_numpy())
    y = np.absolute(table['Shear Stress [MPa]'].to_numpy())
    xnew = np.arange(0, 1, 1e-5)
    ynew = np.interp(xnew, x, y)
    
    
    new_table = pd.DataFrame(columns=('Displacement [mm]', 'Shear Stress [MPa]'))
    new_table['Displacement [mm]'] = xnew
    new_table['Shear Stress [MPa]'] = ynew
    
    return new_table

def filtering_slope(table, var, ax, dictionary):
    
    
    # identifying max value
    table = table[table['Displacement [mm]'] < 0.6]
    max_value = np.argmax(table['Shear Stress [MPa]'])
    table = table[0:max_value]
    
    X = table['Displacement [mm]'][(table['Displacement [mm]'] < 0.4) & (table['Displacement [mm]'] > 0.2)].values.reshape(-1, 1)
    Y = table['Shear Stress [MPa]'][(table['Displacement [mm]'] < 0.4) & (table['Displacement [mm]'] > 0.2)].values.reshape(-1, 1)
    
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
           
    displacement = np.arange(0, 0.6, 0.005).reshape(-1, 1)
    Y_pred = linear_regressor.predict(displacement)  # make predictions
    Y_pred = Y_pred - linear_regressor.intercept_[0]
    
    name = var[1].rstrip(".txt")
    ax.plot(displacement, Y_pred, label = name)
    ax.legend()
    ax.set_ylim(0,60)
    ax.set_xlim(0,1)
    plt.ylabel(ylabel = '$\u03C4_{12}$ [MPa]', fontsize=15)
    plt.xlabel(xlabel = 'Displacement [mm]',fontsize=15)
    
    print(f'Slope = {linear_regressor.coef_[0][0]:.2f} [MPa/mm]')
    
    dictionary[var[0]] = displacement
    dictionary[name] = Y_pred

    return dictionary

def statistical_analysis(address, dictionary):
    min_length = np.min([len(dictionary[0]), len(dictionary[1]), len(dictionary[2]), len(dictionary[3]), len(dictionary[4])])
    
    # initializing a new dataframe
    df = pd.DataFrame()
    
    # extracting x-axis values
    df['Displacement [mm]'] = dictionary[0][0:min_length].reshape(1,-1)[0]
    
    # importing shear stresses from each sample
    df['Force_01'] = dictionary['sample_01'][0:min_length].reshape(1,-1)[0]
    df['Force_02'] = dictionary['sample_02'][0:min_length].reshape(1,-1)[0]
    df['Force_03'] = dictionary['sample_03'][0:min_length].reshape(1,-1)[0]
    df['Force_04'] = dictionary['sample_04'][0:min_length].reshape(1,-1)[0]
    df['Force_05'] = dictionary['sample_05'][0:min_length].reshape(1,-1)[0]
    
    # performing statistical analysis
    df['Average'] = df[['Force_01', 'Force_02', 'Force_03', 'Force_04', 'Force_05']].mean(axis = 1)
    df['Std'] = df[['Force_01', 'Force_02', 'Force_03', 'Force_04', 'Force_05']].std(axis = 1)
    
    return df