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
data = data.drop(['direction_vent', 'ratio_vent_surface_200m_lidar', 'vitesse_zonal', 'vitesse_meridional'], axis = 1) # on se sépare dans le dataset de 2 colonnes
data = data.drop([306, 410, 64, 522], axis = 0) #se séparer des outliers indésirables, discriminés en visualisant les nombres extrèmes du jeu de données
data = data.dropna(axis = 0) # vérification, mais il n'y avait aucune donnée manquante
data.describe() # donne des infos intéressantes (ex : moyenne etc...)





"""Deuxième étape : standardisation des données"""


features = ['Vitesse_vent', 'ratio_vent_surface_200m', 'diff_temp_air_mer', 'flux_vertical_chaleur_surface', 'direction_vent_zonal', 'direction_vent_meridional']
x = data.loc[:, features].values
data_norm = StandardScaler().fit_transform(x) #on norme/centre/réduit la data en utilisant la standartisation de sklearn


for i in range(len(data_norm[0, :])):
    data_norm = np.delete(data_norm, 0 + i, 1)
    
    print(i + 1) #pour voir le nombre de tour de boucle réalisé
    
    
    
    """Troisième étape : Réalisation  de l'ACP"""
    
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(data_norm)
    principaldata = pd.DataFrame(data = principalComponents)
    
    
    
    """Quatrième étape : COMBIEN FAUT-IL DE DIMENSION? & VISUALISATIOn de la PCA"""
    

    plt.figure(figsize = (8, 8))
    plt.plot([1, 2, 3, 4, 5], np.cumsum(pca.explained_variance_ratio_), label = 'Var. cumulé sans ' + features[i])
    plt.xlabel('nombre de dimensions', fontsize = 20)
    plt.ylabel('variance cumulée', fontsize = 20)
    plt.legend(fontsize = 20)
    plt.axhline(linewidth=4, color='r', linestyle = '--', y= 0.7)
    print( "Dans le cas ou on se sépare de la colonne ", features[i],"\n --> La variance des 5 axes est : ", pca.explained_variance_ratio_)
    
    #Fonction pour visualiser la PCA
    #seuil de variance minimum pour considérer une bonne PCA fixé à 0.8
    if float(np.sum(pca.explained_variance_ratio_[0:2])) > 0.8:
        print(np.sum(pca.explained_variance_ratio_[0:2]), 'est bien au dessus du seuil de variance cumulée convenable fixé à 0.7, 2 dimensions suffiront pour bien visualiser la PCA')
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(data_norm)
        principaldata = pd.DataFrame(data = principalComponents)
        plt.figure(figsize = (8,8))
        a = principaldata.loc[:,0]
        b = principaldata.loc[:,1]
        plt.scatter(a, b, s = 10, c = 'red')
        plt.show()
        

    elif float(np.sum(pca.explained_variance_ratio_[0:3])) > 0.7:
        print(np.sum(pca.explained_variance_ratio_[0:3]), 'est bien au dessus du seuil de variance cumulée convenable fixé à 0.7, 3 dimensions suffiront pour bien visualiser la PCA')
        
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(data_norm)
        principaldata = pd.DataFrame(data = principalComponents)
        
        fig = plt.figure(figsize = (8,8))
        a = principaldata.loc[:, 0]
        b = principaldata.loc[:, 1]
        d = principaldata.loc[:, 2]
        ax = fig.add_subplot(111, projection = "3d")
        ax.scatter(a, b, d, c ='red', s = 10 )
        plt.title('Visualisation en 3 dimensions sans ' + features[i] , fontsize = 20)
    
    else :
        print("Meme 3 dimensions ne permettent pas de retourner 80% de l'information, on ne fait pas de PCA")
    
    #retour a la ligne
    print("\n")    
    #on renorme x, pour remettre la colonne qu'on a enlevé et passer au nouveau tour de boucle
    data_norm = StandardScaler().fit_transform(x)
    #on incrément i :
    i += 1
    
plt.show()
















"""

# ----   Pour faire passer la 3eme dimension en couleur si 3 dimensions


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(data_norm)
principaldata = pd.DataFrame(data = principalComponents)

fig = plt.figure(figsize = (8,8))
a = principaldata.loc[:,0]
b = principaldata.loc[:,1]
d = principaldata.loc[:, 2]
plt.scatter(a, b, s = 50, c = d)
plt.colorbar()
"""