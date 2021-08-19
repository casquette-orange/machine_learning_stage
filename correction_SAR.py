# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:54:53 2021

@author: tbeaudelain
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import warnings; warnings.filterwarnings(action='ignore')

from prettytable import PrettyTable
from sklearn.datasets import load_iris
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, ridge_regression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset


"""  --------------------     SOUS PROGRAMMES --------------------------  """


def normalize_angles(direction, azimuth):
    a = 0
    for i in np.arange(0,direction.shape[0]):
        print(a)
        a += 1
        for j in np.arange(0,direction.shape[1]):
            for k in np.arange(0,direction.shape[2]):
                if direction[0,j,k]  < 0:
                    direction[0,j,k]=direction[0,j,k]+360
       
    azimuth_diff=direction+(360-azimuth)-90
    a = 0
    for i in np.arange(0,azimuth_diff.shape[0]):
        print(a)
        a += 1
        for j in np.arange(0,azimuth_diff.shape[1]):
            for k in np.arange(0,azimuth_diff.shape[2]):
                if azimuth_diff[0,j,k]  > 360:
                    azimuth_diff[0,j,k]=azimuth_diff[0,j,k]-360
                elif azimuth_diff[0,j,k]  < 0:
                    azimuth_diff[0,j,k]=azimuth_diff[0,j,k]+360
    return direction, azimuth_diff



"""  --------------------     PROGRAMMES --------------------------  """


# 1/ Affectation des paramètres dans les  variables

datacube = Dataset('sentinel1_wrf_colocs_datacube.nc')

wspd_temp=np.squeeze(datacube.variables['owiWindSpeed'][:])
wspd = wspd_temp.filled(np.nan)

wdir_temp=np.squeeze(datacube.variables['owiWindDirection'][:])
wdir = wdir_temp.filled(np.nan)

azimuth_temp=np.squeeze(datacube.variables['owiHeading'][:])
azimuth = azimuth_temp.filled(np.nan)

elevation_temp=np.squeeze(datacube.variables['owiElevationAngle'][:])
elevation = elevation_temp.filled(np.nan)

incidence_temp=np.squeeze(datacube.variables['owiIncidenceAngle'][:])
incidence = incidence_temp.filled(np.nan)

sigma_temp=np.squeeze(datacube.variables['owiNrcs'][:])
sigma = sigma_temp.filled(np.nan)

time_temp=np.squeeze(datacube.variables['sar_time'][:])
time = time_temp.filled(np.nan)




# 2/ prétraitement:

sigma[sigma==0]=np.nan
sigma=10*np.log10(sigma)
wspd=wspd*((4/10)**0.11)

direction, angle_diff=normalize_angles(wdir_temp, azimuth_temp)
param_list = [wspd, time, direction, angle_diff, incidence, sigma ]








# 3/ Traitement et sauvegarde de input_ML

input_ML=np.zeros((wspd.shape[0],wspd.shape[1],wspd.shape[2],len(param_list)))
for i in np.arange(0,len(param_list)):
        if i==1:
            for j in np.arange(0,wspd.shape[1]):
                for k in np.arange(0,wspd.shape[2]):
                    input_ML[:,j,k,i]=param_list[i]
        else:
            input_ML[:,:,:,i]=param_list[i]
                        
np.save('input_ML.npy', input_ML)
input_ML = np.load('input_ML.npy')
input_ML[:,:,:,0] = input_ML[:,:,:,0]*((4/10)**0.11)








# 4/ Traitement et sauvegarde de input_corrected_ML

from scipy.stats import skewnorm
param=np.load('param_error_low_wind_sentinel1.npy')
with open('model_correction_sentinel1.pkl', 'rb') as f:
    random_forest_correction = pickle.load(f)
input_ML_corrected=np.copy(input_ML)
z = 0
for i in np.arange(input_ML.shape[1]):
    #print('geometric correction ', i)
    print(z)
    z += 1
    for j in np.arange(input_ML.shape[2]):
        rf_input=np.copy(input_ML[:,i,j,:])
        # idx_1=np.where(rf_input[:,0]<1)
        # rf_input[idx_1[0],0]=np.nan
        test_nan=np.sum(rf_input, axis=1)
        idx_nan=np.where(np.isnan(test_nan))
        idx_notnan=np.where(~np.isnan(test_nan))
        idx_low=np.where((rf_input[:,0]<=1.5) & ~np.isnan(test_nan))
        idx_high=np.where((rf_input[:,0]>1.5) & ~np.isnan(test_nan))
        idx_veryhigh=np.where((rf_input[:,0]>13) & ~np.isnan(test_nan))
        input_ML_corrected[idx_nan[0],i,j,0]=np.nan
        rf_input2=np.copy(rf_input[idx_high[0],:])
        #print(len(rf_input2))
        if len(rf_input2>0):
            input_ML_corrected[idx_high[0],i,j,0]=random_forest_correction.predict(rf_input2)
        input_ML_corrected[idx_low[0],i,j,0]=input_ML[idx_low[0],i,j,0]+skewnorm.rvs(param[0], loc=param[1], scale=param[2], size=len(idx_low[0]))
        for k in np.arange(0,len(input_ML_corrected[:,i,j,0])):
            if input_ML_corrected[k,i,j,0] <=0:
                input_ML_corrected[k,i,j,0]=0.5#np.nan
            if input_ML[k,i,j,0] >30:
                input_ML_corrected[k,i,j,0]=np.nan



input_ML_corrected[:,:,:,0] = input_ML_corrected[:,:,:,0]*((10/4)**0.11)
np.save('input_corrected_ML.npy', input_ML_corrected)



    

