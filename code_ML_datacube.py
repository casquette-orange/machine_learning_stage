# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:24:12 2021

@author: tbeaudelain
"""


"""            1 - IMPORTATION DES MODULES          """

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import warnings; warnings.filterwarnings(action='ignore')

from prettytable import PrettyTable
from netCDF4 import Dataset

from sklearn import ensemble
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor, ridge_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fonctions_ML_datacube import *





"""            2 - PRE-PREOCESSING          """





""" Chargement du Datcube + Correction Sar du vent """

# Ouverture du dataset avec Xarray
plt.close('all')
print('preprocessing en cours ...')
DS_new = xr.open_dataset('sentinel1_wrf_colocs_datacube.nc')
# CORRECTION DU VENT SAR
DS_new  = correction_vent_SAR(DS_new)




""" Slicing du datacube avant ML + Transform     Netcdf Object -->  DataFrame Pandas object"""

# Choix du slicing du datacube selon les 4 variables  : x, y, hybrid, time

latitude_debut, latitude_fin = 30, 33
taille_zone= ((latitude_fin - latitude_debut) + 1)
longitude_debut, longitude_fin = 30, 33
temps_debut, temps_fin = 0, 1280



# Sclicing

altitude = 2
DS_date_range = DS_new.sel(x = slice(latitude_debut, latitude_fin))
DS_date_range = DS_date_range.sel(y = slice(longitude_debut, longitude_fin))
DS_date_range = DS_date_range.sel(time =  slice(temps_debut, temps_fin))
DS_date_range = DS_date_range.sel(hybrid = altitude)
df = DS_date_range.to_dataframe()
# Ajout d'un vent à une altitude :
altitude2 = 3
DS_date_range = DS_new.sel(x = slice(latitude_debut, latitude_fin))
DS_date_range = DS_date_range.sel(y = slice(longitude_debut, longitude_fin))
DS_date_range = DS_date_range.sel(time =  slice(temps_debut, temps_fin))
DS_date_range = DS_date_range.sel(hybrid = altitude2)
df40m = DS_date_range.to_dataframe()
vent_40m = df40m['wind_speed']
df['wind_40m'] = vent_40m




#  Transform     DataFrame Pandas object --> Numpy array object


# créer un fichier csv
#df.head()
#df.to_csv("datacube_slice1_67km2.csv", index=True, sep=";")


features = [ 'owiWindSpeed', 'wind_speed', 'temperature', 'water_temperature_surface','relative_humidity', 'wind_dir', 'sensible_heat_net_flux_surface', 'wind_speed']#, 'longitude', 'latitude', 'wind_20m', 'wind_100m','longitude', 'latitude', 'surface_wind_gust', 'wind_40m', 'wind_200m' ]
x = df.loc[:, features].values


#Fonction pour modifier la dernière colonne --> pour faire le vent moyen de l'image à traiter
a = 0
for i in range(0, int(len(x)/(taille_zone**2))):
    moyenne = np.mean(x[(a, a+(taille_zone**2 - 1)),7])
    for i in range(0,(taille_zone**2) ):
        x[(a+i),7] = moyenne
    a = a + (taille_zone**2)
        

#Moyenne de touutes les features
# a = 0
# for j in range(2, 8):
#     a = 0
#     for i in range(0, int(len(x)/(taille_zone**2))):
#         moyenne = np.mean(x[(a, a+(taille_zone**2 - 1)),j])
#         for i in range(0,(taille_zone**2) ):
#             x[(a+i),j] = moyenne
#         a = a + (taille_zone**2)





# Code pour pratiquer un Shuffle et enlever les lignes avec des N/A
x = np.reshape(x, (temps_fin - temps_debut, ((taille_zone)**2)*len(features)))
rng = np.random.default_rng()
rng.shuffle(x)
for i in range(0, 10):
    np.random.shuffle(x)
x = pd.DataFrame(x)
x.dropna(inplace = True)
x = x.to_numpy()
x = np.reshape(x, (x.shape[0]*((taille_zone)**2), len(features)))


# x = np.delete(x, 0, axis = 1)
#x = x[0::16,:]

# graphe pour visualiser le shuffle
def Vent_SAR_sur_5_ans(x):
    plt.figure(3)
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.plot(np.arange(0, int(len(x))), x[:,1])
    plt.xlabel('Vitesse cible du vent SAR (m/s)', fontsize=20)
    plt.ylabel('Vitesses (modèle / ML) (m/s)', fontsize=20)
    plt.legend()
    plt.show()
    
#Vent_SAR_sur_5_ans(x)




# Nettoyage des colonnes avant l'apprentissage
""" calcul densité
longitude = x[:,2] - 360
pressure = x[:,8]
humidity = x[:,7]
temperature = x[:,4] - 273.15
humidity=humidity/100
density=1/(287.06*(temperature + 273.15))*(pressure-230.617*humidity*np.exp((17.5043*temperature)/(241.2+temperature)))
x[:,2] = longitude
x[:,4] = density
x[:,6] = np.sin((np.pi/180)*x[:,6])
x[:,9] = np.cos((np.pi/180)*x[:,9]) 


# x = np.delete(x,2, axis = 1)
# x = np.delete(x,2, axis = 1)
# x = np.delete(x,3, axis = 1)
# x = np.delete(x,6, axis = 1)
"""

y = x[:,0]
x = np.delete(x, 0, axis = 1)


""" Code pour prendre un pixel sur 2
dataa = x
y_long = dataa[:,0]
y = y_long[0::2]
y = y[0::2]
y = y[0::2]
x_long = np.delete(dataa,0, axis = 1)
x = x_long[0::2,:]
x = x[0::2,:]
x = x[0::2,:]
"""



"""            3 - APPEL DES SOUS-PROGRAMMES         """

from fonctions_ML_datacube import *




""" SOUS PROGRAMMES - TRAITEMENT DONNES / APPRENTISSAGE """
#SOUS PROGRAMME PCA
#x = pca(x, features)
#SOUS PROGRAMME ENTRAINEMENT
vitesse, vitesse_ML, y_test, random_forest, vitesse_train, y_train = train_ML_correction(x, y)
#SOUS PROGRAMME BILAN STAT
stats_results_ML(vitesse,vitesse_ML,y_test,latitude_debut, latitude_fin)


""" SOUS PROGRAMMES PLOTTING """
plt.figure(1)
plt.rcParams['figure.figsize'] = (8, 12)
#SOUS PROGRAMME TRACER HISTOGRAMME IMPORTANCES PARAMETRES
hist_plot_features(random_forest)
#SOUS PROGRAMME TRACER HISTOGRAMME
hist_plot(y_test, vitesse_ML, vitesse)
#SOUS PROGRAMME TRACER COURBE
courbe_plot(y_test, vitesse_ML, vitesse, taille_zone)


#SOUS PROGRAMME POUR CARTE ERREURS
plot_carte(y_test, vitesse_ML, vitesse, vitesse_train, y_train, taille_zone)
#SOUS PROGRAMME PLOTXY
plot_x_y(y_test, vitesse_ML, vitesse)















