# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:19:29 2021

@author: tbeaudelain
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd







# on affecte juste à y la liste de sindex de à à 968
y = data.loc[:,['Vitesse_vent']].index



# CENTRER ET REDUIRE LES DONNEES
from sklearn.preprocessing import StandardScaler
data_norm = StandardScaler().fit_transform(x)



# METHODE GAUSSIAN KDE DU MODULE STATS DE SPICY
from scipy.stats import gaussian_kde



# suppresion des outliers(broadcasting), car il ya des ratios de vents excessifs
z = finaldata.loc[:, 'ratio_vent_surface_200m_lidar']
z = finaldata.loc[:, 'ratio_vent_surface_200m_lidar'].where(finaldata.loc[:,'ratio_vent_surface_200m_lidar'] < 2, inplace  = True)
z = finaldata.loc[:, 'ratio_vent_surface_200m_lidar'].where(finaldata.loc[:,'ratio_vent_surface_200m_lidar'] > 0.5)



# REALISER UN SCATTERPLOT 3D
from mpl_toolkits.mplot3d import Axes3d
ax = plt.axes(projection='3d')
ax.scatter(data['Vitesse_vent'], data['ratio_vent_surface_200m' ])



# REALISER UN PAIRPLOT AVEC SEABORN POUR VISUALISER GLOBALEME?T LES DONNEES
import seaborn as sns
sns.pairplot(data, hue = 'Vitesse_vent')



"""         ACP           """



# REALISATION DUNE ACP            
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_norm)
principaldata = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])















