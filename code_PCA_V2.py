# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:19:29 2021

@author: tbeaudelain
"""





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D







"""Première étape : chargement des données"""


data = pd.read_csv('dataset_classification_simple.csv', delimiter = ";") #importation des données
data = data.drop([306, 410, 64, 522], axis = 0) #se séparer des outliers indésirables, discriminés en visualisant les nombres extrèmes du jeu de données
X = data.to_numpy()[:,1]
data = data.drop(['direction_vent', 'ratio_vent_surface_200m_lidar'], axis = 1) # on se sépare dans le dataset de 2 colonnes
data = data.dropna(axis = 0) # vérification, mais il n'y avait aucune donnée manquante
data.describe() # donne des infos intéressantes (ex : moyenne etc...)





"""Deuxième étape : standardisation des données"""


features = ['Vitesse_vent', 'ratio_vent_surface_200m', 'diff_temp_air_mer', 'flux_vertical_chaleur_surface', 'direction_vent_zonal', 'direction_vent_meridional']
x = data.loc[:, features].values
data_norm = StandardScaler().fit_transform(x) #on norme/centre/réduit la data en utilisant la standartisation de sklearn


fig = plt.figure(figsize = (15, 10))



for i in range(len(data_norm[0, :])):
    z = data.to_numpy()[:,i]
    #if i == 5:
    #    z = X
    

    """Réalisation  de l'ACP"""
    
    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(data_norm)
    principaldata = pd.DataFrame(data = principalComponents)
    
    
    a = principaldata.loc[:, 0]
    b = principaldata.loc[:, 1]
    d = principaldata.loc[:, 2]
    ax = fig.add_subplot(2, 3, i+1, projection = "3d")
    t = ax.scatter(a, b, d, c = z, s = 10)
    plt.title('couleur =' + features[i])
    fig.colorbar(t)
    ax.legend()

    
    print("\n")    
    data_norm = StandardScaler().fit_transform(x) #on renorme x, pour remettre la colonne qu'on a enlevé et passer au nouveau tour de boucle
    i += 1
    
    
print(sum(pca.explained_variance_ratio_[0:3]))
plt.show()

evr = pca.explained_variance_ratio_
cvr = np.cumsum(pca.explained_variance_ratio_)

pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr

pca_dims = []
for x in range(0, len(pca_df)):
    pca_dims.append('PCA Component {}'.format(x))
pca=  pd.DataFrame(pca.components_, columns=features, index=pca_dims)
print(pca.head(6).T)


